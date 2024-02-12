import random
import numpy as np
import torch


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
