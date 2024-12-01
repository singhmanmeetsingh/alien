import numpy as np
import torch
import random
from collections import namedtuple, deque
from numpy.random import choice

Experience = namedtuple("Experience", field_names=[
                        "state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(
            np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones), None, None

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, device):
        super().__init__(action_size, buffer_size, batch_size, device)
        self.priorities = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        self.priorities.append(max(self.priorities, default=1.0))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance_weights(self, probabilities, beta):
        importance = (1/len(self.memory) * 1/probabilities) ** beta
        importance_normalized = importance / max(importance)
        return torch.tensor(importance_normalized, dtype=torch.float32).to(self.device)

    def sample(self, priority_scale=1.0, beta=0.4):
        sample_size = min(self.batch_size, len(self.memory))
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = choice(range(len(self.memory)),
                                sample_size, p=sample_probs, replace=False)

        experiences = [self.memory[idx] for idx in sample_indices]
        importance_weights = self.get_importance_weights(
            sample_probs[sample_indices], beta)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(
            np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones), importance_weights, sample_indices

    def update_priorities(self, indices, errors, offset=0.1):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = float(abs(error.item()) + offset)
