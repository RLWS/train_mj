from __future__ import annotations

import torch

from mjai.features.action_features import RESPONSE_ACTION_DIM, TURN_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM
from mjai.models.multitask_policy_net import MultitaskMahjongPolicyNet


def test_multitask_policy_net_forward_shapes():
    model = MultitaskMahjongPolicyNet(hidden_size=32, dropout=0.0)

    turn_batch = {
        "state_feats": torch.zeros(2, TURN_STATE_DIM),
        "candidate_feats": torch.zeros(2, 4, TURN_ACTION_DIM),
        "candidate_mask": torch.tensor(
            [
                [True, True, True, False],
                [True, False, False, False],
            ]
        ),
    }
    response_batch = {
        "state_feats": torch.zeros(3, RESPONSE_STATE_DIM),
        "candidate_feats": torch.zeros(3, 5, RESPONSE_ACTION_DIM),
        "candidate_mask": torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, False, False, False, False],
            ]
        ),
    }

    turn_logits = model.forward_turn(turn_batch)
    response_logits = model.forward_response(response_batch)

    assert turn_logits.shape == (2, 4)
    assert response_logits.shape == (3, 5)
    assert turn_logits[0, 3].item() == -1e9
    assert response_logits[2, 1].item() == -1e9
