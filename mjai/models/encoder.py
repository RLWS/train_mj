from __future__ import annotations

import torch
from torch import nn


class MahjongEncoder(nn.Module):
    """Small shared encoder for state and candidate action features."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, dropout: float) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            state_dim (int): 函数执行所需的输入参数。
            action_dim (int): 函数执行所需的输入参数。
            hidden_size (int): 隐藏层维度。
            dropout (float): 随机失活比例。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(
        self,
        state_feats: torch.Tensor,
        candidate_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # state_feats: [B, state_dim] -> [B, H]
        # candidate_feats: [B, C, action_dim] -> [B, C, H]; nn.Linear applies
        # over the last dimension and preserves the candidate axis.
        """执行模型前向传播。
        
        参数:
            self (Self): 当前对象实例。
            state_feats (torch.Tensor): 状态特征张量。
            candidate_feats (torch.Tensor): 候选动作特征张量。
        
        返回:
            tuple[torch.Tensor, torch.Tensor]: 元组结果。
        
        异常:
            无。
        """
        state_hidden = self.state_encoder(state_feats)
        candidate_hidden = self.action_encoder(candidate_feats)
        return state_hidden, candidate_hidden
