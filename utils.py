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
    Overlay normalized actions on the state using a single operation.
    Actions are normalized from [-1, 1] to [0, 1], and non-zero values replace corresponding values in state.

    Parameters:
    - state: A tensor representing the state.
    - action: A tensor with the same shape as state, where values in [-1, 1] represent actions to be overlaid.

    Returns:
    - A new tensor where the state is modified by overlaying normalized action values.
    """
    # Normalize action values from [-1, 1] to [0, 1]
    normalized_action = (action + 1) / 2

    # Create a mask of non-zero (action) values after normalization
    action_mask = (
        normalized_action != 0.5
    )  # 0.5 corresponds to the original action value of 0

    # Use the mask to select where to overlay normalized action values onto the state
    state[action_mask] = normalized_action[action_mask]

    return state
