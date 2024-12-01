"""Intrinsic Curiosity Module (ICM) models

This module implements the three main components of the ICM architecture:
1. Encoder: Converts raw states into feature representations
2. Forward Model: Predicts next state features given current state and action
3. Inverse Model: Predicts the action taken between two states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderModel(nn.Module):
    """State encoder for ICM that converts raw observations into feature representations"""

    def __init__(self, input_channels=4):
        """Initialize encoder network

        Args:
            input_channels (int): Number of input channels (e.g., 4 for stacked frames)
        """
        super(EncoderModel, self).__init__()

        # Convolutional layers process the input frames
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        # Initialize weights using He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization for better training"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through encoder

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Encoded state representation
        """
        x = F.normalize(x)  # Normalize input to improve training stability
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        return x.flatten(start_dim=1)  # Flatten spatial dimensions


class InverseModel(nn.Module):
    """Inverse dynamics model that predicts actions between states"""

    def __init__(self, encoded_size, action_size):
        """Initialize inverse model

        Args:
            encoded_size (int): Size of encoded state representation
            action_size (int): Size of action space
        """
        super(InverseModel, self).__init__()

        # Fully connected layers to predict actions
        # *2 because we concatenate two states
        self.linear1 = nn.Linear(encoded_size * 2, 256)
        self.linear2 = nn.Linear(256, action_size)

    def forward(self, state1, state2):
        """Forward pass through inverse model

        Args:
            state1 (torch.Tensor): Encoded current state
            state2 (torch.Tensor): Encoded next state

        Returns:
            torch.Tensor: Predicted action probabilities
        """
        x = torch.cat((state1, state2),
                      dim=1)  # Concatenate state representations
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.softmax(x, dim=1)  # Convert to action probabilities


class ForwardModel(nn.Module):
    """Forward dynamics model that predicts next state features"""

    def __init__(self, encoded_size, action_size):
        """Initialize forward model

        Args:
            encoded_size (int): Size of encoded state representation
            action_size (int): Size of action space
        """
        super(ForwardModel, self).__init__()

        # Calculate input size: encoded state + one-hot action
        input_size = encoded_size + action_size

        # Fully connected layers to predict next state
        self.linear1 = nn.Linear(input_size, 256)
        # Output size matches encoded state size
        self.linear2 = nn.Linear(256, encoded_size)

    def forward(self, state, action):
        """Forward pass through forward model

        Args:
            state (torch.Tensor): Encoded current state
            action (torch.Tensor): Action tensor

        Returns:
            torch.Tensor: Predicted next state encoding
        """
        # Convert action to one-hot representation
        device = state.device
        action_one_hot = torch.zeros(action.shape[0], 18).to(device)
        indices = torch.stack(
            (torch.arange(action.shape[0]).to(device), action.squeeze()), dim=0)
        action_one_hot[indices.tolist()] = 1.0

        # Concatenate state and action
        x = torch.cat((state, action_one_hot), dim=1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


def loss_fn(q_loss, inverse_loss, forward_loss, beta=0.2, lambda_=0.1):
    """Combined loss function for ICM

    This function combines three components:
    1. Q-learning loss: Main task objective
    2. Forward model loss: Prediction error for next state
    3. Inverse model loss: Prediction error for actions

    Args:
        q_loss (torch.Tensor): Q-learning loss
        inverse_loss (torch.Tensor): Inverse model loss
        forward_loss (torch.Tensor): Forward model loss
        beta (float): Weight for forward loss vs inverse loss (0-1)
        lambda_ (float): Weight for ICM loss vs Q-learning loss

    Returns:
        torch.Tensor: Combined loss value
    """
    # Combine forward and inverse losses with beta weighting
    icm_loss = (1 - beta) * inverse_loss + beta * forward_loss
    icm_loss = icm_loss.sum() / icm_loss.flatten().shape[0]

    # Combine ICM loss with Q-learning loss using lambda weighting
    total_loss = lambda_ * q_loss + icm_loss

    return total_loss
