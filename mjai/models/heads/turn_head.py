from __future__ import annotations

import torch
from torch import nn


class TurnHead(nn.Module):
    """Score legal turn candidates from encoded state and action features."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            hidden_size (int): 隐藏层维度。
            dropout (float): 随机失活比例。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        super().__init__()
        joint_size = hidden_size * 4
        self.scorer = nn.Sequential(
            nn.LayerNorm(joint_size),
            nn.Linear(joint_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        state_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Broadcast one encoded state to every legal candidate, then score each
        # candidate-state pair independently.
        """执行模型前向传播。
        
        参数:
            self (Self): 当前对象实例。
            state_hidden (torch.Tensor): 函数执行所需的输入参数。
            candidate_hidden (torch.Tensor): 函数执行所需的输入参数。
            candidate_mask (torch.Tensor): 候选动作掩码张量。
        
        返回:
            torch.Tensor: 张量结果。
        
        异常:
            无。
        """
        expanded_state = state_hidden.unsqueeze(1).expand_as(candidate_hidden)
        joint = torch.cat(
            [
                expanded_state,
                candidate_hidden,
                expanded_state * candidate_hidden,
                torch.abs(expanded_state - candidate_hidden),
            ],
            dim=-1,
        )
        logits = self.scorer(joint).squeeze(-1)
        # Padding candidates are impossible choices and must not participate in
        # argmax or cross entropy.
        return logits.masked_fill(~candidate_mask, -1e9)
