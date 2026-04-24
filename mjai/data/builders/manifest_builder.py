from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq

from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class ManifestSummary:
    train_games: int
    valid_games: int
    test_games: int
    train_turn_samples: int
    valid_turn_samples: int
    test_turn_samples: int
    train_response_samples: int
    valid_response_samples: int
    test_response_samples: int
    warnings: tuple[str, ...] = ()


class ManifestBuilder:
    """Create simple game-level manifests from parquet shard directories."""

    def build(self, input_dir: str | Path, output_dir: str | Path) -> ManifestSummary:
        input_root = Path(input_dir)
        output_root = ensure_dir(output_dir)

        turn_files = sorted((input_root / "turn").glob("*.parquet"))
        game_ids = [path.stem for path in turn_files]

        train_ids, valid_ids, test_ids = _split_game_ids(game_ids)
        split_sample_totals: dict[str, dict[str, int]] = {}
        warnings: list[str] = []

        for split_name, ids in [("train", train_ids), ("valid", valid_ids), ("test", test_ids)]:
            records = []
            turn_rows_total = 0
            response_rows_total = 0
            for game_id in ids:
                turn_path = input_root / "turn" / f"{game_id}.parquet"
                response_path = input_root / "response" / f"{game_id}.parquet"
                turn_rows = pq.read_metadata(turn_path).num_rows
                response_rows = pq.read_metadata(response_path).num_rows
                records.append(
                    {
                        "game_id": game_id,
                        "turn_path": str(turn_path.as_posix()),
                        "turn_rows": turn_rows,
                        "response_path": str(response_path.as_posix()),
                        "response_rows": response_rows,
                    }
                )
                turn_rows_total += turn_rows
                response_rows_total += response_rows
            (output_root / f"{split_name}.jsonl").write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            split_sample_totals[split_name] = {
                "turn_samples": turn_rows_total,
                "response_samples": response_rows_total,
            }
            if not ids:
                warnings.append(f"split '{split_name}' has 0 games")
            if turn_rows_total == 0:
                warnings.append(f"split '{split_name}' has 0 turn samples")
            if response_rows_total == 0:
                warnings.append(f"split '{split_name}' has 0 response samples")

        summary = ManifestSummary(
            train_games=len(train_ids),
            valid_games=len(valid_ids),
            test_games=len(test_ids),
            train_turn_samples=split_sample_totals["train"]["turn_samples"],
            valid_turn_samples=split_sample_totals["valid"]["turn_samples"],
            test_turn_samples=split_sample_totals["test"]["turn_samples"],
            train_response_samples=split_sample_totals["train"]["response_samples"],
            valid_response_samples=split_sample_totals["valid"]["response_samples"],
            test_response_samples=split_sample_totals["test"]["response_samples"],
            warnings=tuple(warnings),
        )
        (output_root / "manifest_summary.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary


def _split_game_ids(game_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    train_ids: list[str] = []
    valid_ids: list[str] = []
    test_ids: list[str] = []

    for game_id in sorted(game_ids):
        bucket = int(hashlib.md5(game_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket < 80:
            train_ids.append(game_id)
        elif bucket < 90:
            valid_ids.append(game_id)
        else:
            test_ids.append(game_id)
    return train_ids, valid_ids, test_ids
