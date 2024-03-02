import torch
from torchvision.utils import save_image as torchvision_save_image

import random
import numpy as np

import os


def set_random_seed(seed=2021, deterministic=False):
    """
    모든 메이저 랜덤 넘버 생성 라이브러리에 대해 시드를 설정하는 함수.

    Args:
    seed (int): 사용할 시드 값.
    deterministic (bool): True로 설정하면 CuDNN을 사용하는 작업에서 재현 가능한 결과를 얻을 수 있음.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_image(tensor, filepath):
    """Save a tensor as an image, ensuring directories are created."""
    # Ensure the parent directory exists
    ensure_dir(os.path.dirname(filepath))
    # Normalize the tensor to [0, 1] if it's not already
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # Use torchvision's save_image function
    torchvision_save_image(tensor, filepath)


def ensure_dir(path):
    """Ensure that a directory exists; if not, create it."""
    os.makedirs(path, exist_ok=True)


def overlay_actions_on_state(state, action):
    """
    Overlay actions on the state using a single operation.
    Non-zero values in action replace corresponding values in state.

    Parameters:
    - state: A tensor representing the state.
    - action: A tensor with the same shape as state, where non-zero values are actions to be overlaid.

    Returns:
    - A new tensor where the state is modified by overlaying action values.
    """
    # Create a mask of non-zero (action) values
    action_mask = action != 0

    # Use the mask to select where to overlay action values onto the state
    state[action_mask] = action[action_mask]

    return state
