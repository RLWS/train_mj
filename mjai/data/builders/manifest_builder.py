from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class ManifestSummary:
    train_games: int
    valid_games: int
    test_games: int


class ManifestBuilder:
    """Create simple game-level manifests from parquet shard directories."""

    def build(self, input_dir: str | Path, output_dir: str | Path) -> ManifestSummary:
        input_root = Path(input_dir)
        output_root = ensure_dir(output_dir)

        turn_files = sorted((input_root / "turn").glob("*.parquet"))
        game_ids = [path.stem for path in turn_files]

        train_ids, valid_ids, test_ids = _split_game_ids(game_ids)

        for split_name, ids in [("train", train_ids), ("valid", valid_ids), ("test", test_ids)]:
            records = []
            for game_id in ids:
                turn_path = input_root / "turn" / f"{game_id}.parquet"
                response_path = input_root / "response" / f"{game_id}.parquet"
                records.append(
                    {
                        "game_id": game_id,
                        "turn_path": str(turn_path.as_posix()),
                        "turn_rows": pq.read_metadata(turn_path).num_rows,
                        "response_path": str(response_path.as_posix()),
                        "response_rows": pq.read_metadata(response_path).num_rows,
                    }
                )
            (output_root / f"{split_name}.jsonl").write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )

        return ManifestSummary(
            train_games=len(train_ids),
            valid_games=len(valid_ids),
            test_games=len(test_ids),
        )


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
