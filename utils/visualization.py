"""Utilities for visualizing training results"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def plot_scores(scores, window_size=100, title="Training Scores"):
    """Plot training scores with moving average

    Args:
        scores (list): List of episode scores
        window_size (int): Size of moving average window
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))

    # Plot raw scores
    plt.plot(scores, alpha=0.3, color='blue', label='Raw Scores')

    # Calculate and plot moving average
    moving_avg = np.convolve(scores, np.ones(
        window_size)/window_size, mode='valid')
    plt.plot(np.arange(window_size-1, len(scores)), moving_avg, color='red',
             label=f'{window_size}-Episode Moving Average')

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def save_training_plots(scores, config, save_path):
    """Save training visualization plots

    Args:
        scores (list): List of episode scores
        config (dict): Training configuration
        save_path (str): Path to save plots
    """
    # Create main score plot
    fig = plot_scores(scores,
                      title=f"Training Scores - {
                          'DQN' if config['dqn'] else 'DDQN'} "
                      f"{'with' if config['prio'] else 'without'} PER "
                      f"{'with' if config['icm'] else 'without'} ICM")

    # Save plot
    fig.savefig(save_path)
    plt.close(fig)


def create_comparison_plot(score_sets, labels, window_size=100):
    """Create comparison plot of multiple training runs

    Args:
        score_sets (list): List of score lists from different runs
        labels (list): Labels for each run
        window_size (int): Size of moving average window
    """
    plt.figure(figsize=(12, 6))

    for scores, label in zip(score_sets, labels):
        moving_avg = np.convolve(scores, np.ones(
            window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(scores)),
                 moving_avg, label=label)

    plt.title('Training Score Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()
