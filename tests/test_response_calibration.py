from __future__ import annotations

import torch

from mjai.eval.response_calibration import _ResponseCalibrationScanner
from mjai.eval.response_error_sampling import _select_response_predictions
from mjai.features.action_features import RESPONSE_ACTION_DIM
from mjai.training.metrics import RESPONSE_KIND_RANK


def test_response_calibration_tracks_no_yaku_false_call_rate() -> None:
    batch = _batch_for_no_yaku_false_call()
    logits = torch.tensor([[0.50, 0.75]], dtype=torch.float32)
    scanner = _ResponseCalibrationScanner(
        pass_biases=[0.0],
        no_yaku_call_penalties=[0.0, 0.4],
    )

    scanner.update(logits=logits, batch=batch)
    points = scanner.points()

    raw_point = points[0]
    safe_point = points[1]
    assert safe_point.equivalent_call_margin == 0.0
    assert safe_point.no_yaku_equivalent_call_margin == 0.4
    assert raw_point.false_call_count == 1
    assert raw_point.no_yaku_false_call_count == 1
    assert raw_point.no_yaku_false_call_rate == 1.0
    assert raw_point.no_yaku_false_call_share == 1.0
    assert safe_point.false_call_count == 0
    assert safe_point.no_yaku_false_call_count == 0
    assert safe_point.missed_call_count == 0


def test_response_error_sampling_applies_no_yaku_call_penalty() -> None:
    batch = _batch_for_no_yaku_false_call()
    logits = torch.tensor([[0.50, 0.75]], dtype=torch.float32)

    raw = _select_response_predictions(logits=logits, batch=batch, pass_bias=0.0)
    safe = _select_response_predictions(
        logits=logits,
        batch=batch,
        pass_bias=0.0,
        no_yaku_call_penalty=0.4,
    )

    assert int(raw.item()) == 1
    assert int(safe.item()) == 0


def _batch_for_no_yaku_false_call() -> dict:
    candidate_feats = torch.zeros((1, 2, RESPONSE_ACTION_DIM), dtype=torch.float32)
    candidate_feats[0, 1, 48 + 38] = 1.0
    return {
        "labels": torch.tensor([0], dtype=torch.long),
        "candidate_mask": torch.tensor([[True, True]]),
        "candidate_counts": torch.tensor([2], dtype=torch.long),
        "label_kind_ids": torch.tensor([RESPONSE_KIND_RANK["pass"]], dtype=torch.long),
        "candidate_feats": candidate_feats,
        "legal_actions": [[{"kind": "pass"}, {"kind": "chi", "tile": "3m"}]],
    }
