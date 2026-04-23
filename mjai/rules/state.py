from __future__ import annotations

from dataclasses import dataclass, field

from mjai.rules.tiles import AKA_BASE_INDICES, base_index_to_tile, parse_tile, tile_to_base_index

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


@dataclass(slots=True, frozen=True)
class MeldState:
    """Minimal meld state needed for replay validation."""

    kind: str
    source: int
    called_tile: str
    consumed_tiles: tuple[str, ...]

    @property
    def base_index(self) -> int:
        return tile_to_base_index(self.called_tile)


@dataclass(slots=True)
class GameState:
    """Four-player replay state for parsed Tenhou events."""

    hands: list[HandState] = field(default_factory=lambda: [HandState() for _ in range(4)])
    discards: list[list[str]] = field(default_factory=lambda: [[] for _ in range(4)])
    melds: list[list[MeldState]] = field(default_factory=lambda: [[] for _ in range(4)])
    riichi_declared: list[bool] = field(default_factory=lambda: [False] * 4)
    last_discard_actor: int | None = None
    last_discard_tile: str | None = None
    hand_index: int = -1

    @classmethod
    def from_init_payload(cls, hand_index: int, payload: dict) -> "GameState":
        hands = [
            HandState.from_tiles(payload["hai0"]),
            HandState.from_tiles(payload["hai1"]),
            HandState.from_tiles(payload["hai2"]),
            HandState.from_tiles(payload["hai3"]),
        ]
        return cls(hands=hands, hand_index=hand_index)

    def apply_draw(self, actor: int, tile: str) -> None:
        self.hands[actor].add_tile(tile)

    def apply_discard(self, actor: int, tile: str) -> None:
        self.hands[actor].remove_tile(tile)
        self.discards[actor].append(tile)
        self.last_discard_actor = actor
        self.last_discard_tile = tile

    def apply_reach(self, actor: int, step: int) -> None:
        if step == 1:
            self.riichi_declared[actor] = True

    def apply_meld(
        self,
        actor: int,
        kind: str,
        source: int,
        called_tile: str,
        consumed_tiles: list[str],
    ) -> None:
        if kind in {"chi", "pon", "daiminkan"}:
            if self.last_discard_actor != source:
                raise ValueError(
                    f"meld source mismatch: actor={actor} source={source} "
                    f"last_discard_actor={self.last_discard_actor}"
                )
            if self.last_discard_tile != called_tile:
                raise ValueError(
                    f"called tile mismatch: expected last discard {self.last_discard_tile}, got {called_tile}"
                )
            for tile in consumed_tiles:
                self.hands[actor].remove_tile(tile)
            self.melds[actor].append(
                MeldState(
                    kind=kind,
                    source=source,
                    called_tile=called_tile,
                    consumed_tiles=tuple(consumed_tiles),
                )
            )
            self.last_discard_actor = None
            self.last_discard_tile = None
            return

        if kind == "ankan":
            for tile in consumed_tiles:
                self.hands[actor].remove_tile(tile)
            self.melds[actor].append(
                MeldState(
                    kind=kind,
                    source=source,
                    called_tile=called_tile,
                    consumed_tiles=tuple(consumed_tiles),
                )
            )
            return

        if kind == "kakan":
            self._upgrade_pon_to_kakan(actor=actor, called_tile=called_tile, consumed_tiles=consumed_tiles)
            return

        if kind == "nuki":
            return

        raise ValueError(f"unsupported meld kind: {kind}")

    def _upgrade_pon_to_kakan(self, actor: int, called_tile: str, consumed_tiles: list[str]) -> None:
        if not consumed_tiles:
            raise ValueError("kakan is missing consumed tiles")

        added_tile = consumed_tiles[-1]
        self.hands[actor].remove_tile(added_tile)
        base_index = tile_to_base_index(called_tile)

        for meld_index, meld in enumerate(self.melds[actor]):
            if meld.kind == "pon" and meld.base_index == base_index:
                upgraded = MeldState(
                    kind="kakan",
                    source=meld.source,
                    called_tile=meld.called_tile,
                    consumed_tiles=meld.consumed_tiles + (added_tile,),
                )
                self.melds[actor][meld_index] = upgraded
                return

        raise ValueError(f"kakan without existing pon: actor={actor} tile={called_tile}")
