"""Helper functions for the Alien RL project"""

import os
import random
import torch
import numpy as np
import pickle
from collections import deque


def seed_everything(seed):
    """Set random seeds for reproducibility

    Args:
        seed (int): Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def moving_average(data, window_size=100):
    """Calculate moving average of data

    Args:
        data (list): Input data
        window_size (int): Size of moving window

    Returns:
        list: Moving averages
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def create_checkpoint_name(config):
    """Create a unique checkpoint name based on configuration

    Args:
        config (dict): Training configuration

    Returns:
        str: Unique checkpoint name
    """
    components = [
        str(config['seed']),
        str(config['state_size']),
        str(config['action_size']),
        'DQN' if config['dqn'] else 'DDQN',
        'PRIO' if config['prio'] else 'STD',
        'ICM' if config['icm'] else 'NO-ICM'
    ]
    return '_'.join(components)


def save_checkpoint(agent, memory, scores_window, config, episode, checkpoint_dir='saved_models'):
    """Save training checkpoint

    Args:
        agent: The training agent
        memory: Replay buffer
        scores_window: Recent scores
        config (dict): Training configuration
        episode (int): Current episode number
        checkpoint_dir (str): Directory for saving checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = create_checkpoint_name(config)
    path = os.path.join(checkpoint_dir, f"{checkpoint_name}_ep{episode}")

    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.qnetwork_local.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'scores_window': scores_window,
        'config': config
    }

    if hasattr(agent, 'qnetwork_target'):
        checkpoint['target_state_dict'] = agent.qnetwork_target.state_dict()

    torch.save(checkpoint, f"{path}_model.pth")

    # Save memory if using prioritized replay
    if config['prio']:
        with open(f"{path}_memory.pkl", 'wb') as f:
            pickle.dump(memory, f)


def load_checkpoint(path, agent, config):
    """Load training checkpoint

    Args:
        path (str): Path to checkpoint
        agent: The training agent
        config (dict): Training configuration

    Returns:
        tuple: (episode number, scores window)
    """
    checkpoint = torch.load(path + "_model.pth")

    agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if hasattr(agent, 'qnetwork_target'):
        agent.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])

    if config['prio']:
        with open(path + "_memory.pkl", 'rb') as f:
            memory = pickle.load(f)
            return checkpoint['episode'], checkpoint['scores_window'], memory

    return checkpoint['episode'], checkpoint['scores_window'], None
