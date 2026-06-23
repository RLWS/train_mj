from __future__ import annotations

import pytest

from mjai.training.metrics import ResponseMetricsAccumulator, combine_response_diagnostics


def test_combine_response_diagnostics_uses_counts():
    first = ResponseMetricsAccumulator()
    first.update(
        label_kind="pass",
        predicted_kind="chi",
        candidate_kinds=["pass", "chi"],
        candidate_count=2,
        label_rank=2,
    )
    first.update(
        label_kind="chi",
        predicted_kind="chi",
        candidate_kinds=["pass", "chi"],
        candidate_count=2,
        label_rank=1,
    )

    second = ResponseMetricsAccumulator()
    second.update(
        label_kind="pass",
        predicted_kind="pass",
        candidate_kinds=["pass"],
        candidate_count=1,
        label_rank=1,
    )

    combined = combine_response_diagnostics([first.summary(), second.summary()])

    assert combined is not None
    assert combined.single_candidate.samples == 1
    assert combined.multi_candidate.samples == 2
    assert combined.multi_candidate.top1_accuracy == 0.5
    assert combined.multi_candidate.top3_accuracy == 1.0
    assert combined.binary.actual_pass == 2
    assert combined.binary.actual_non_pass == 1
    assert combined.binary.false_call_count == 1
    assert combined.binary.false_call_rate == 0.5
    assert combined.binary.non_pass_f1 == pytest.approx(2.0 / 3.0)
    assert combined.confusion == {
        "chi->chi": 1,
        "pass->chi": 1,
        "pass->pass": 1,
    }
