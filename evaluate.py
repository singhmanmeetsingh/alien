"""Evaluation script for trained Alien RL models"""

import argparse
import os
import torch
import numpy as np
from collections import deque

from models.agent import Agent
from utils.environment import (
    setup_environment,
    initialize_state_deque,
    create_state_stack,
    update_state_deque
)
from utils.helpers import seed_everything
from loggers.tensorboard_logger import TensorboardLogger
from config.hyperparameters import DEFAULT_CONFIG  # Import default configuration


def load_model(checkpoint_path, agent):
    """Load a trained model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path)
        agent.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(agent, 'qnetwork_target'):
            agent.qnetwork_target.load_state_dict(
                checkpoint['target_state_dict'])
        print(f"Successfully loaded model from {checkpoint_path}")
        return checkpoint['config']
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")


def evaluate_episode(env, agent, config, render=False):
    """Run a single evaluation episode"""
    state, _ = env.reset(seed=config['seed'])
    state_deque = initialize_state_deque(state, config['stack_size'])
    current_state = create_state_stack(state_deque)

    episode_reward = 0
    episode_steps = 0
    done = False

    while not done and episode_steps < config['max_t']:
        # Select action (no exploration during evaluation)
        action = agent.act(current_state, eps=0.0)

        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Process next state
        state_deque = update_state_deque(state_deque, next_state)
        next_state_tensor = create_state_stack(state_deque)

        # Update state and metrics
        current_state = next_state_tensor
        episode_reward += reward
        episode_steps += 1

        if render:
            env.render()

    return episode_reward, episode_steps


def run_evaluation(checkpoint_path, num_episodes=100, render=False):
    """Run full evaluation of a trained model"""
    # Setup environment
    env = setup_environment(render_mode='human' if render else 'rgb_array')

    # Create temporary agent with default configuration
    temp_config = DEFAULT_CONFIG.copy()
    temp_config.update({
        'dqn': False,
        'prio': False,
        'icm': False,
        'name': 'Temporary Agent'  # Add a name for logging
    })

    temp_agent = Agent(
        state_size=temp_config['state_size'],
        action_size=temp_config['action_size'],
        config=temp_config
    )

    # Load the actual configuration from checkpoint
    config = load_model(checkpoint_path, temp_agent)

    # Create proper agent with loaded config
    agent = Agent(
        state_size=config['stack_size'],
        action_size=config['action_size'],
        config=config
    )
    load_model(checkpoint_path, agent)

    # Initialize logger for evaluation
    logger = TensorboardLogger(config, log_dir="eval_logs")

    # Tracking metrics
    scores = []
    steps_list = []
    scores_window = deque(maxlen=100)

    print("\nStarting evaluation...")
    print("=" * 50)
    print("Episode    Steps    Score     Avg Score")
    print("-" * 50)

    try:
        for i_episode in range(1, num_episodes + 1):
            score, steps = evaluate_episode(env, agent, config, render)

            # Update tracking metrics
            scores.append(score)
            steps_list.append(steps)
            scores_window.append(score)

            # Log metrics
            logger.log_episode(
                episode=i_episode,
                score=score,
                eps=0.0,  # No exploration during evaluation
                avg_score=np.mean(scores_window),
                memory_len=0,  # Not using memory during evaluation
                steps=steps,
                total_steps=sum(steps_list)
            )

            # Print progress
            print(f"\r{i_episode:7d} {steps:8d} {score:9.1f} {
                  np.mean(scores_window):11.1f}", end="")

            if i_episode % 10 == 0:
                print(f"\nLast 10 episodes: Mean: {np.mean(scores[-10:]):.1f}, "
                      f"Min: {np.min(scores[-10:]):.1f}, Max: {np.max(scores[-10:]):.1f}")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    finally:
        env.close()
        logger.close()

    # Print final statistics
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"Number of episodes: {len(scores)}")
    print(f"Average score: {np.mean(scores):.1f}")
    print(f"Standard deviation: {np.std(scores):.1f}")
    print(f"Minimum score: {np.min(scores):.1f}")
    print(f"Maximum score: {np.max(scores):.1f}")
    print(f"Average steps per episode: {np.mean(steps_list):.1f}")

    return scores, steps_list


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained Alien RL agent')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    seed_everything(args.seed)

    # Run evaluation
    try:
        scores, steps = run_evaluation(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            render=args.render
        )

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"\nEvaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
