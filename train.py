"""Main training script with runtime configuration selection"""

import numpy as np
from collections import deque
import torch
from utils.environment import (
    setup_environment,
    initialize_state_deque,
    create_state_stack,
    update_state_deque
)
from utils.helpers import seed_everything
from models.agent import Agent
from loggers.tensorboard_logger import TensorboardLogger
from config.config_selector import create_selected_config, print_config_summary
from utils.helpers import save_checkpoint, create_checkpoint_name


def initialize_training(config):
    """Initialize all components needed for training"""
    try:
        env = setup_environment()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize environment: {str(e)}")

    seed_everything(config['seed'])
    logger = TensorboardLogger(config)

    agent = Agent(
        state_size=config['stack_size'],
        action_size=config['action_size'],
        config=config
    )

    return agent, env, logger


def train(config):
    """Main training function implementing the full training loop with progress display"""
    # Initialize training components
    agent, env, logger = initialize_training(config)

    # Initialize tracking variables
    scores = []
    scores_window = deque(maxlen=config['scores_len'])
    eps = config['eps_start']
    total_steps = 0

    print("\nStarting training with configuration:", config['name'])
    print("=" * 60)
    print("Episode  Steps    Epsilon    Score     Avg Score    Memory Size")
    print("-" * 60)

    try:
        # Main training loop
        for i_episode in range(1, config['n_episodes'] + 1):
            # Run episode
            episode_steps = 0
            state, _ = env.reset(seed=config['seed'])
            state_deque = initialize_state_deque(state, config['stack_size'])
            current_state = create_state_stack(state_deque)
            score = 0

            # Episode loop
            for t in range(config['max_t']):
                # Select and perform action
                action = agent.act(current_state, eps)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Process next state
                state_deque = update_state_deque(state_deque, next_state)
                next_state_tensor = create_state_stack(state_deque)

                # Store experience and learn
                agent.step(current_state, action, reward,
                           next_state_tensor, done)

                # Update state and metrics
                current_state = next_state_tensor
                score += reward
                episode_steps += 1
                total_steps += 1

                if done:
                    break

            # Update tracking metrics
            scores_window.append(score)
            scores.append(score)
            eps = max(config['eps_end'], config['eps_decay'] * eps)

            # Print progress
            if i_episode % 1 == 0:  # Print every episode
                print(f"\r{i_episode:7d} {episode_steps:7d} {eps:9.3f} {score:9.1f} {
                      np.mean(scores_window):11.1f} {len(agent.memory):12d}", end="")

            # Save checkpoint based on save_interval
            if config['save'] and i_episode % config['save_interval'] == 0:
                save_checkpoint(
                    agent=agent,
                    memory=agent.memory,
                    scores_window=scores_window,
                    config=config,
                    episode=i_episode,
                    checkpoint_dir=config['checkpoint_dir']
                )
                print(f"\nCheckpoint saved at episode {i_episode}")

            # Detailed progress update every N episodes
            if i_episode % config['scores_len'] == 0:
                print(f"\n{'-' * 60}")  # Add separator line
                print(f"Episode {i_episode} completed:")
                print(f"Average Score: {np.mean(scores_window):.2f}")
                print(f"Total Steps: {total_steps}")
                print(f"Current Epsilon: {eps:.3f}")
                print(f"Memory Size: {len(agent.memory)}")
                print(f"{'-' * 60}")

            # Log episode metrics
            logger.log_episode(
                episode=i_episode,
                score=score,
                eps=eps,
                avg_score=np.mean(scores_window),
                memory_len=len(agent.memory),
                losses=agent.get_latest_losses()
            )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Completed {i_episode} episodes, {total_steps} total steps")

        # Save final checkpoint on interrupt
        if config['save']:
            save_checkpoint(
                agent=agent,
                memory=agent.memory,
                scores_window=scores_window,
                config=config,
                episode=i_episode,
                checkpoint_dir=config['checkpoint_dir']
            )
            print("Final checkpoint saved")

    except Exception as e:
        print(f"\n\nTraining failed with error: {str(e)}")
        raise
    finally:
        logger.close()
        env.close()
        print("\nTraining completed!")
        print(f"Final Average Score: {np.mean(scores_window):.2f}")
        print(f"Total Steps: {total_steps}")

    return agent, scores

# def train(config):
#     """Main training function implementing the full training loop with progress display"""
#     # Initialize training components
#     agent, env, logger = initialize_training(config)
#
#     # Initialize tracking variables
#     scores = []
#     scores_window = deque(maxlen=config['scores_len'])
#     eps = config['eps_start']
#     total_steps = 0
#
#     print("\nStarting training with configuration:", config['name'])
#     print("=" * 60)
#     print("Episode  Steps    Epsilon    Score     Avg Score    Memory Size")
#     print("-" * 60)
#
#     try:
#         # Main training loop
#         for i_episode in range(1, config['n_episodes'] + 1):
#             # Run episode
#             episode_steps = 0
#             state, _ = env.reset(seed=config['seed'])
#             state_deque = initialize_state_deque(state, config['stack_size'])
#             current_state = create_state_stack(state_deque)
#             score = 0
#
#             # Episode loop
#             for t in range(config['max_t']):
#                 # Select and perform action
#                 action = agent.act(current_state, eps)
#                 next_state, reward, terminated, truncated, _ = env.step(action)
#                 done = terminated or truncated
#
#                 # Process next state
#                 state_deque = update_state_deque(state_deque, next_state)
#                 next_state_tensor = create_state_stack(state_deque)
#
#                 # Store experience and learn
#                 agent.step(current_state, action, reward,
#                            next_state_tensor, done)
#
#                 # Update state and metrics
#                 current_state = next_state_tensor
#                 score += reward
#                 episode_steps += 1
#                 total_steps += 1
#
#                 if done:
#                     break
#
#             # Update tracking metrics
#             scores_window.append(score)
#             scores.append(score)
#             eps = max(config['eps_end'], config['eps_decay'] * eps)
#
#             # Print progress
#             if i_episode % 1 == 0:  # Print every episode
#                 print(f"\r{i_episode:7d} {episode_steps:7d} {eps:9.3f} {score:9.1f} {
#                       np.mean(scores_window):11.1f} {len(agent.memory):12d}", end="")
#
#             # Detailed progress update every N episodes
#             if i_episode % config['scores_len'] == 0:
#                 print(f"\n{'-' * 60}")  # Add separator line
#                 print(f"Episode {i_episode} completed:")
#                 print(f"Average Score: {np.mean(scores_window):.2f}")
#                 print(f"Total Steps: {total_steps}")
#                 print(f"Current Epsilon: {eps:.3f}")
#                 print(f"Memory Size: {len(agent.memory)}")
#                 print(f"{'-' * 60}")
#
#             # Log episode metrics
#             logger.log_episode(
#                 episode=i_episode,
#                 score=score,
#                 eps=eps,
#                 avg_score=np.mean(scores_window),
#                 memory_len=len(agent.memory),
#                 losses=agent.get_latest_losses(),
#                 steps=episode_steps,
#                 total_steps=total_steps
#             )
#
#     except KeyboardInterrupt:
#         print("\n\nTraining interrupted by user")
#         print(f"Completed {i_episode} episodes, {total_steps} total steps")
#     except Exception as e:
#         print(f"\n\nTraining failed with error: {str(e)}")
#         raise
#     finally:
#         logger.close()
#         env.close()
#         print("\nTraining completed!")
#         print(f"Final Average Score: {np.mean(scores_window):.2f}")
#         print(f"Total Steps: {total_steps}")
#
#     return agent, scores
#
#


def main():
    # Get configuration from user
    config = create_selected_config()

    # Display selected configuration
    print_config_summary(config)

    # Confirm with user
    confirm = input("\nProceed with training? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled.")
        return

    try:
        print("\nStarting training...")
        agent, scores = train(config)
        print(f"\nTraining completed!")
        print(f"Final average score: {np.mean(scores[-100:]):.2f}")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")


if __name__ == "__main__":
    main()
