import gymnasium as gym
import numpy as np
from skimage.transform import resize
import torch
import ale_py
from collections import deque


def setup_environment(render_mode='rgb_array'):
    gym.register_envs(ale_py)
    env = gym.make('ALE/Alien-v5', render_mode=render_mode)
    return env


def downscale_observation(observation, new_size=(84, 84), to_gray=True):
    """Process raw frames from the environment"""
    frame = observation
    if to_gray and len(frame.shape) == 3:
        # Convert to grayscale and normalize
        frame = frame.mean(axis=2) / 255.0
    resized_frame = resize(frame, new_size, anti_aliasing=True)
    return resized_frame


def prepare_state(state):
    """Convert a single state to proper tensor format"""
    processed_state = downscale_observation(state, to_gray=True)
    # Shape: (1, 1, 84, 84)
    return torch.from_numpy(processed_state).float().unsqueeze(0).unsqueeze(0)


def prepare_multi_state(state1, state2):
    """Prepare consecutive states for stacking"""
    state1 = state1.clone()
    processed_state2 = downscale_observation(state2, to_gray=True)
    tmp = torch.from_numpy(processed_state2).float()

    # Roll the frames in the stack (4 frames total)
    state1[0, 0] = state1[0, 1]
    state1[0, 1] = state1[0, 2]
    state1[0, 2] = state1[0, 3]
    state1[0, 3] = tmp

    return state1


def prepare_initial_state(state, N=4):
    """Prepare initial state with N stacked copies"""
    processed_state = downscale_observation(state, to_gray=True)
    state_tensor = torch.from_numpy(processed_state).float()

    # Create initial stack - Shape: (1, N, 84, 84)
    stacked_state = torch.stack([state_tensor for _ in range(N)], dim=0)
    stacked_state = stacked_state.unsqueeze(0)

    return stacked_state


def create_state_stack(state_deque):
    """Convert deque of frames to proper tensor format"""
    # Stack frames along channel dimension - Shape: (1, 4, 84, 84)
    stacked_frames = torch.stack(list(state_deque), dim=0)
    return stacked_frames.unsqueeze(0)


def initialize_state_deque(state, maxlen=4):
    """Initialize deque with processed initial frame copies"""
    processed_state = downscale_observation(state, to_gray=True)
    state_tensor = torch.from_numpy(processed_state).float()

    state_deque = deque(maxlen=maxlen)
    for _ in range(maxlen):
        state_deque.append(state_tensor)

    return state_deque


def update_state_deque(state_deque, new_state):
    """Update deque with new processed frame"""
    processed_state = downscale_observation(new_state, to_gray=True)
    state_tensor = torch.from_numpy(processed_state).float()
    state_deque.append(state_tensor)
    return state_deque


def get_state_shape():
    """Get the shape of processed state for network initialization"""
    return (84, 84)  # Height, Width of processed frames
