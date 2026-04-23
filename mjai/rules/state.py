from __future__ import annotations

from dataclasses import dataclass, field

from mjai.rules.tiles import AKA_BASE_INDICES, base_index_to_tile, parse_tile

AKA_TILES_BY_INDEX = {
    4: "0m",
    13: "0p",
    22: "0s",
}


@dataclass(slots=True)
class HandState:
    """Minimal hand state for local rule validation and legal discard generation."""

    counts: list[int] = field(default_factory=lambda: [0] * 34)
    aka_counts: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_tiles(cls, tiles: list[str]) -> "HandState":
        hand = cls()
        for tile in tiles:
            hand.add_tile(tile)
        return hand

    def total_tiles(self) -> int:
        return sum(self.counts)

    def count_exact(self, tile: str) -> int:
        parsed = parse_tile(tile)
        if parsed.is_aka:
            return self.aka_counts.get(parsed.base_index, 0)
        if parsed.base_index in AKA_BASE_INDICES:
            return self.counts[parsed.base_index] - self.aka_counts.get(parsed.base_index, 0)
        return self.counts[parsed.base_index]

    def add_tile(self, tile: str) -> None:
        parsed = parse_tile(tile)
        self.counts[parsed.base_index] += 1
        if parsed.is_aka:
            self.aka_counts[parsed.base_index] = self.aka_counts.get(parsed.base_index, 0) + 1

    def remove_tile(self, tile: str) -> None:
        parsed = parse_tile(tile)
        if self.count_exact(tile) <= 0:
            raise ValueError(f"tile not available in hand: {tile}")

        self.counts[parsed.base_index] -= 1
        if parsed.is_aka:
            remaining = self.aka_counts.get(parsed.base_index, 0) - 1
            if remaining < 0:
                raise ValueError(f"red tile bookkeeping underflow: {tile}")
            if remaining == 0:
                self.aka_counts.pop(parsed.base_index, None)
            else:
                self.aka_counts[parsed.base_index] = remaining

    def has_tile(self, tile: str) -> bool:
        return self.count_exact(tile) > 0

    def snapshot(self) -> dict[str, int]:
        """Return an exact-tile multiset for debugging and tests."""
        tiles: dict[str, int] = {}
        for index, count in enumerate(self.counts):
            if count <= 0:
                continue
            red_count = self.aka_counts.get(index, 0)
            if red_count:
                tiles[AKA_TILES_BY_INDEX[index]] = red_count
            normal_count = count - red_count
            if normal_count > 0:
                tiles[base_index_to_tile(index)] = normal_count
        return tiles
