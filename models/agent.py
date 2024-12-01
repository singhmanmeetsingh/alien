import torch
import torch.nn.functional as F
import numpy as np
import random

from models.q_network import QNetwork
from models.icm_models import EncoderModel, ForwardModel, InverseModel, loss_fn
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class Agent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize Q-Networks with proper input channels for frame stacking
        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        if not config['icm']:
            self.qnetwork_target = QNetwork(
                state_size, action_size).to(self.device)
            self.qnetwork_target.load_state_dict(
                self.qnetwork_local.state_dict())

        # Initialize ICM if enabled
        if config['icm']:
            self.encoder = EncoderModel(
                input_channels=state_size).to(self.device)
            self.forward_model = ForwardModel(288, action_size).to(self.device)
            self.inverse_model = InverseModel(288, action_size).to(self.device)

            parameters = (list(self.qnetwork_local.parameters()) +
                          list(self.encoder.parameters()) +
                          list(self.forward_model.parameters()) +
                          list(self.inverse_model.parameters()))
        else:
            parameters = self.qnetwork_local.parameters()

        self.optimizer = torch.optim.Adam(parameters, lr=config['lr'])

        # Initialize replay memory with proper buffer configuration
        if config['prio']:
            self.memory = PrioritizedReplayBuffer(
                action_size, config['buffer_size'],
                config['batch_size'], self.device
            )
        else:
            self.memory = ReplayBuffer(
                action_size, config['buffer_size'],
                config['batch_size'], self.device
            )

        self.t_step = 0
        self.latest_losses = {}

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.config['update_every']
        if self.t_step == 0 and len(self.memory) > self.config['batch_size']:
            experiences, importance_weights, indices = self.memory.sample()
            self.learn(experiences, importance_weights, indices)

    def act(self, state, eps=0.):
        """Select action using epsilon-greedy policy with proper state handling"""
        # Handle different input types and ensure correct shape
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        if not isinstance(state, torch.Tensor):
            raise TypeError("State must be either numpy array or torch tensor")

        # Add batch dimension if missing
        if state.dim() == 3:
            state = state.unsqueeze(0)

        # Move to correct device
        state = state.to(self.device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, importance_weights=None, indices=None):
        states, actions, rewards, next_states, dones = experiences

        if self.config['icm']:
            self.learn_with_icm(states, actions, rewards, next_states, dones)
        else:
            self.learn_standard(states, actions, rewards, next_states, dones,
                                importance_weights, indices)

    def learn_standard(self, states, actions, rewards, next_states, dones,
                       importance_weights, indices):
        # Handle Double DQN vs regular DQN
        if self.config['dqn']:
            Q_targets_next = self.qnetwork_target(
                next_states).detach().max(1)[0].unsqueeze(1)
        else:
            actions_next = self.qnetwork_local(
                next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(
                next_states).detach().gather(1, actions_next)

        Q_targets = rewards + \
            (self.config['gamma'] * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Handle prioritized replay if enabled
        if importance_weights is not None:
            td_errors = Q_expected - Q_targets
            loss = torch.mean((td_errors ** 2) * importance_weights)
            self.memory.update_priorities(indices, td_errors)
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        self.latest_losses['q_loss'] = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not self.config['icm']:
            self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def learn_with_icm(self, states, actions, rewards, next_states, dones):
        # Your existing ICM implementation remains the same
        # The rest of your ICM implementation...
        encoded_states = self.encoder(states)
        encoded_next_states = self.encoder(next_states)

        predicted_next_states = self.forward_model(encoded_states, actions)
        forward_loss = F.mse_loss(
            predicted_next_states, encoded_next_states.detach(), reduction='none')

        predicted_actions = self.inverse_model(
            encoded_states, encoded_next_states)
        inverse_loss = F.cross_entropy(
            predicted_actions, actions.squeeze(), reduction='none')

        with torch.no_grad():
            intrinsic_reward = torch.mean(forward_loss, dim=1, keepdim=True)

        qvals_next = self.qnetwork_local(next_states)
        combined_reward = intrinsic_reward + 0.2 * \
            torch.max(qvals_next, dim=1, keepdim=True)[0]

        q_values = self.qnetwork_local(states)
        q_targets = q_values.clone()

        batch_indices = torch.arange(actions.shape[0], device=self.device)
        q_targets[batch_indices, actions.squeeze()] = combined_reward.squeeze()

        q_loss = F.mse_loss(F.normalize(q_values),
                            F.normalize(q_targets.detach()))

        total_loss = loss_fn(
            q_loss=q_loss,
            inverse_loss=inverse_loss,
            forward_loss=forward_loss,
            beta=self.config.get('icm_beta', 0.2),
            lambda_=self.config.get('icm_lambda', 0.1)
        )

        self.latest_losses.update({
            'q_loss': q_loss.item(),
            'forward_loss': forward_loss.mean().item(),
            'inverse_loss': inverse_loss.mean().item(),
            'total_loss': total_loss.item()
        })

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model):
        tau = self.config['tau']
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_latest_losses(self):
        return self.latest_losses

    # Save and load methods remain the same...
    def save(self, path):
        # Your existing save implementation...
        save_dict = {
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            't_step': self.t_step
        }

        if not self.config['icm']:
            save_dict['qnetwork_target_state_dict'] = self.qnetwork_target.state_dict()
        else:
            save_dict.update({
                'encoder_state_dict': self.encoder.state_dict(),
                'forward_model_state_dict': self.forward_model.state_dict(),
                'inverse_model_state_dict': self.inverse_model.state_dict()
            })

        torch.save(save_dict, path)

    def load(self, path):
        # Your existing load implementation...
        checkpoint = torch.load(path)

        self.qnetwork_local.load_state_dict(
            checkpoint['qnetwork_local_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.t_step = checkpoint['t_step']

        if not self.config['icm']:
            self.qnetwork_target.load_state_dict(
                checkpoint['qnetwork_target_state_dict'])
        else:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.forward_model.load_state_dict(
                checkpoint['forward_model_state_dict'])
            self.inverse_model.load_state_dict(
                checkpoint['inverse_model_state_dict'])
