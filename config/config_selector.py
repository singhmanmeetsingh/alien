"""Configuration selector for runtime training mode selection"""

from typing import Dict, Optional
import sys
from config.hyperparameters import DEFAULT_CONFIG


def get_variant_groups():
    """Define groups of training variants"""
    return {
        'Basic': [
            {'name': 'Basic DQN', 'dqn': True, 'prio': False, 'icm': False},
            {'name': 'Basic DDQN', 'dqn': False, 'prio': False, 'icm': False},
        ],
        'PER Variants': [
            {'name': 'DQN + PER', 'dqn': True, 'prio': True, 'icm': False},
            {'name': 'DDQN + PER', 'dqn': False, 'prio': True, 'icm': False},
        ],
        'ICM Variants': [
            {'name': 'DQN + ICM', 'dqn': True, 'prio': False, 'icm': True},
            {'name': 'DDQN + ICM', 'dqn': False, 'prio': False, 'icm': True},
        ],
        'Combined': [
            {'name': 'DQN + PER + ICM', 'dqn': True, 'prio': True, 'icm': True},
            {'name': 'DDQN + PER + ICM', 'dqn': False, 'prio': True, 'icm': True},
        ],
        'Learning Rate Variants': [
            {'name': 'DDQN + PER + ICM (Low LR)', 'dqn': False,
             'prio': True, 'icm': True, 'lr': 0.01},
            {'name': 'DDQN + PER + ICM (High LR)', 'dqn': False,
             'prio': True, 'icm': True, 'lr': 0.1},
        ],
        'Exploration Variants': [
            {
                'name': 'DDQN + PER (Slow Decay)',
                'dqn': False, 'prio': True, 'icm': False,
                'eps_decay': 0.999, 'eps_end': 0.05
            },
            {
                'name': 'DDQN + PER (Fast Decay)',
                'dqn': False, 'prio': True, 'icm': False,
                'eps_decay': 0.99, 'eps_end': 0.01
            },
        ],
        'Memory Variants': [
            {
                'name': 'DDQN + PER (Large Buffer)',
                'dqn': False, 'prio': True, 'icm': False,
                'buffer_size': 200000, 'batch_size': 64
            },
            {
                'name': 'DDQN + PER (Small Buffer)',
                'dqn': False, 'prio': True, 'icm': False,
                'buffer_size': 50000, 'batch_size': 16
            },
        ],
    }


def display_variants():
    """Display available training variants in a formatted way"""
    variant_groups = get_variant_groups()

    print("\nAvailable Training Configurations:")
    print("="*50)

    current_index = 1
    index_map = {}

    for group_name, variants in variant_groups.items():
        print(f"\n{group_name}:")
        print("-"*30)

        for variant in variants:
            print(f"{current_index}. {variant['name']}")
            index_map[current_index] = variant
            current_index += 1

    return index_map


def get_user_selection() -> Optional[Dict]:
    """Get user selection for training configuration

    Returns:
        Selected configuration dict or None if selection is invalid
    """
    index_map = display_variants()

    try:
        selection = input(
            "\nEnter the number of the configuration you want to run (or 'q' to quit): ")

        if selection.lower() == 'q':
            print("Exiting...")
            sys.exit(0)

        selection_num = int(selection)
        if selection_num in index_map:
            return index_map[selection_num]
        else:
            print("Invalid selection number!")
            return None

    except ValueError:
        print("Please enter a valid number!")
        return None


def create_selected_config() -> Dict:
    """Create configuration based on user selection

    Returns:
        Complete configuration dictionary
    """
    while True:
        selected_variant = get_user_selection()
        if selected_variant is not None:
            config = DEFAULT_CONFIG.copy()
            config.update(selected_variant)
            return config


def print_config_summary(config: Dict):
    """Print a summary of the selected configuration"""
    print("\nSelected Configuration Summary:")
    print("="*50)
    print(f"Name: {config['name']}")
    print(f"DQN: {'Yes' if config['dqn'] else 'No (DDQN)'}")
    print(f"Prioritized Experience Replay: {
          'Yes' if config['prio'] else 'No'}")
    print(f"Intrinsic Curiosity Module: {'Yes' if config['icm'] else 'No'}")

    # Print special parameters if they exist
    if 'lr' in config:
        print(f"Learning Rate: {config['lr']}")
    if 'eps_decay' in config:
        print(f"Epsilon Decay: {config['eps_decay']}")
    if 'buffer_size' in config:
        print(f"Buffer Size: {config['buffer_size']}")
    print("="*50)
