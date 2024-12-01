"""Configuration parameters for the Alien RL training"""

DEFAULT_CONFIG = {
    # Environment settings
    'seed': 42,
    'state_size': 4,
    'action_size': 18,
    'stack_size': 4,

    # Training parameters
    'n_episodes': 1000,
    'max_t': 2000,
    'scores_len': 50,

    # Memory settings
    'buffer_size': 100000,
    'batch_size': 32,

    # Learning parameters
    'gamma': 0.99,  # discount factor
    'tau': 0.1,     # soft update parameter
    'lr': 0.05,     # learning rate
    'update_every': 4,

    # Exploration settings
    'eps_start': 1.0,
    'eps_end': 0.01,
    'eps_decay': 0.995,

    # Algorithm settings
    'dqn': False,   # True for DQN, False for DDQN
    'prio': True,   # True for Prioritized Experience Replay
    'icm': False,   # True for Intrinsic Curiosity Module

    # Save/Load settings
    'load': False,
    'save': True,

    'save_interval': 2,  # save model every episode
    'checkpoint_dir': 'saved_models'
}
