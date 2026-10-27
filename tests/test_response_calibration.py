from __future__ import annotations

from pathlib import Path

import torch

from mjai.eval.offline import _EvalPartial, _combine_eval_partials, _summary_from_partial
from mjai.eval.response_calibration import (
    _ResponseCalibrationScanner,
    _merge_calibration_scanners,
    calibrate_response_checkpoint,
)
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


def test_response_calibration_reuses_eval_loader_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict = {}

    class FakeDataset:
        def __len__(self) -> int:
            return 1

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs) -> None:
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs

        def __len__(self) -> int:
            return 0

        def __iter__(self):
            return iter(())

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state_dict) -> None:
            pass

        def eval(self) -> None:
            pass

    ckpt_path = tmp_path / "best.pt"
    ckpt_path.write_text("placeholder", encoding="utf-8")
    manifest_path = tmp_path / "valid.jsonl"
    manifest_path.write_text("placeholder\n", encoding="utf-8")

    monkeypatch.setattr(
        "mjai.eval.response_calibration.torch.load",
        lambda *args, **kwargs: {
            "config": {
                "task": {"task": "response"},
                "train": {
                    "device": "cpu",
                    "valid_num_workers": 1,
                    "prefetch_factor": 3,
                    "persistent_workers": True,
                },
                "eval": {
                    "num_workers": 2,
                    "pin_memory": True,
                    "prefetch_factor": 5,
                },
            },
            "feature_schema": "full_state_v2",
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        "mjai.eval.response_calibration._build_calibration_dataset",
        lambda *args, **kwargs: FakeDataset(),
    )
    monkeypatch.setattr("mjai.eval.response_calibration.DataLoader", FakeDataLoader)
    monkeypatch.setattr("mjai.eval.response_calibration.MahjongPolicyNet", lambda **kwargs: FakeModel())

    calibrate_response_checkpoint(
        ckpt_path=ckpt_path,
        split="valid",
        pass_biases=[0.0],
        false_call_rate_targets=[1.0],
        no_yaku_call_penalties=[0.0],
        report_path=tmp_path / "calibration.json",
        manifest_path=manifest_path,
    )

    loader_kwargs = captured["kwargs"]
    assert loader_kwargs["num_workers"] == 2
    assert loader_kwargs["pin_memory"] is True
    assert loader_kwargs["prefetch_factor"] == 5
    assert loader_kwargs["persistent_workers"] is True


def test_response_calibration_scanners_merge_global_counts() -> None:
    batch = _batch_for_no_yaku_false_call()
    logits = torch.tensor([[0.50, 0.75]], dtype=torch.float32)
    first = _ResponseCalibrationScanner(
        pass_biases=[0.0],
        no_yaku_call_penalties=[0.0, 0.4],
    )
    second = _ResponseCalibrationScanner(
        pass_biases=[0.0],
        no_yaku_call_penalties=[0.0, 0.4],
    )
    full = _ResponseCalibrationScanner(
        pass_biases=[0.0],
        no_yaku_call_penalties=[0.0, 0.4],
    )

    first.update(logits=logits, batch=batch)
    second.update(logits=logits, batch=batch)
    full.update(logits=logits.repeat(2, 1), batch=_repeat_batch(batch, 2))

    merged_points = _merge_calibration_scanners([first, second]).points()
    full_points = full.points()

    assert [point.samples for point in merged_points] == [point.samples for point in full_points]
    assert [point.false_call_count for point in merged_points] == [
        point.false_call_count for point in full_points
    ]
    assert [point.no_yaku_false_call_count for point in merged_points] == [
        point.no_yaku_false_call_count for point in full_points
    ]


def test_eval_partials_merge_exact_counts(tmp_path: Path) -> None:
    first = _eval_partial(
        total_loss=3.0,
        samples=3,
        top1_hits=2,
        kind="pass",
        bucket="1-4",
    )
    second = _eval_partial(
        total_loss=5.0,
        samples=2,
        top1_hits=1,
        kind="chi",
        bucket="5-8",
    )

    combined = _combine_eval_partials([first, second], max_misclassified_examples=10)
    summary = _summary_from_partial(
        partial=combined,
        task_name="response",
        split="valid",
        manifest_path=tmp_path / "valid.jsonl",
        checkpoint_path=tmp_path / "best.pt",
        resolved_report_path=tmp_path / "eval.json",
        device=torch.device("cpu"),
    )

    assert summary.samples == 5
    assert summary.batches == 2
    assert summary.loss == 1.6
    assert summary.top1_accuracy == 0.6
    assert summary.per_kind["pass"].samples == 3
    assert summary.per_kind["chi"].samples == 2
    assert summary.candidate_buckets["1-4"].samples == 3
    assert summary.candidate_buckets["5-8"].samples == 2


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


def _repeat_batch(batch: dict, count: int) -> dict:
    return {
        "labels": batch["labels"].repeat(count),
        "candidate_mask": batch["candidate_mask"].repeat(count, 1),
        "candidate_counts": batch["candidate_counts"].repeat(count),
        "label_kind_ids": batch["label_kind_ids"].repeat(count),
        "candidate_feats": batch["candidate_feats"].repeat(count, 1, 1),
        "legal_actions": batch["legal_actions"] * count,
    }


def _eval_partial(
    *,
    total_loss: float,
    samples: int,
    top1_hits: int,
    kind: str,
    bucket: str,
) -> _EvalPartial:
    return _EvalPartial(
        total_loss=total_loss,
        total_samples=samples,
        batches=1,
        topk_hits={1: top1_hits, 3: samples, 5: samples},
        kind_totals={kind: samples},
        kind_loss_totals={kind: total_loss},
        kind_hits={kind: {1: top1_hits, 3: samples, 5: samples}},
        group_totals={kind: samples},
        group_loss_totals={kind: total_loss},
        group_hits={kind: {1: top1_hits, 3: samples, 5: samples}},
        bucket_totals={bucket: samples},
        bucket_hits={bucket: {1: top1_hits, 3: samples, 5: samples}},
        response=None,
        misclassified_examples=[],
    )
