from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """设置 Python、NumPy 与可用深度学习后端的随机种子。
    
    参数:
        seed (int): 随机种子。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
