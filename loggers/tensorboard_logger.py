"""TensorBoard logging functionality for the Alien RL agent"""

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, config, log_dir="logs"):
        """Initialize the TensorBoard logger with enhanced metrics"""
        # Create unique run name based on timestamp and configuration
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Add configuration name
        run_name += f"_{config['name'].replace(' ', '_')}"

        self.log_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(self.log_dir)

        # Create config dict with string values for hparams
        hparam_dict = {}
        for k, v in config.items():
            if isinstance(v, (int, float, bool, str)):
                hparam_dict[k] = v
            else:
                hparam_dict[k] = str(v)

        # Log hyperparameters
        self.writer.add_hparams(
            hparam_dict,
            {'dummy': 0}  # Required placeholder metric
        )

    def log_episode(self, episode, score, eps, avg_score, memory_len, losses=None, steps=None, total_steps=None):
        """Log episode-level metrics with enhanced tracking"""
        # Basic metrics
        self.writer.add_scalar('Score/Episode', score, episode)
        self.writer.add_scalar('Score/Average', avg_score, episode)
        self.writer.add_scalar('Exploration/Epsilon', eps, episode)
        self.writer.add_scalar('Memory/Size', memory_len, episode)

        # Step tracking
        if steps is not None:
            self.writer.add_scalar('Steps/Episode', steps, episode)
        if total_steps is not None:
            self.writer.add_scalar('Steps/Total', total_steps, episode)

        # Learning metrics
        if losses:
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(
                    f'Loss/{loss_name}', loss_value, episode)

        # Calculate steps per episode
        if steps is not None:
            self.writer.add_scalar(
                'Performance/StepsPerEpisode', steps, episode)

        # Calculate score per step
        if steps is not None and steps > 0:
            self.writer.add_scalar(
                'Performance/ScorePerStep', score/steps, episode)

    def log_step(self, step, reward, action=None, q_value=None):
        """Log step-level metrics"""
        self.writer.add_scalar('Step/Reward', reward, step)
        if q_value is not None:
            self.writer.add_scalar('Step/Q-Value', q_value, step)
        if action is not None:
            self.writer.add_scalar('Step/Action', action, step)

    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

    def get_log_dir(self):
        """Get the log directory path"""
        return self.log_dir
