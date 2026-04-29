from __future__ import annotations

from dataclasses import dataclass

from mjai.rules.tiles import tile136_to_string


@dataclass(frozen=True, slots=True)
class Meld:
    """Decoded Tenhou meld event."""

    kind: str
    actor: int
    source: int
    called_tile: str
    consumed_tiles: tuple[str, ...]
    raw_value: int


def decode_meld(actor: int, value: int) -> Meld:
    """Decode Tenhou's packed meld integer.

    Tenhou encodes the relative source player in the low two bits:
    0 = self, 1 = player on the right, 2 = opposite, 3 = player on the left.
    The exact bit layout differs by meld type. This decoder currently supports
    chi, pon, added kan, daiminkan, and ankan in the standard four-player log.
    """
    source = (actor + (value & 0x3)) % 4

    if value & 0x4:
        return _decode_chi(actor, source, value)
    if value & 0x18:
        return _decode_pon_or_kakan(actor, source, value)
    if value & 0x20:
        return _decode_nuki(actor, source, value)
    return _decode_kan(actor, source, value)


def _decode_chi(actor: int, source: int, value: int) -> Meld:
    base_and_called = value >> 10
    sequence_index = base_and_called // 3
    suit = sequence_index // 7
    start = sequence_index % 7

    called_offset = base_and_called % 3
    copies = [(value >> 3) & 0x3, (value >> 5) & 0x3, (value >> 7) & 0x3]

    tile_ids = []
    for offset, copy_index in enumerate(copies):
        base = (suit * 9 + start + offset) * 4
        tile_ids.append(base + copy_index)

    called_tile = tile136_to_string(tile_ids[called_offset])
    consumed_tiles = tuple(
        tile136_to_string(tile_id)
        for offset, tile_id in enumerate(tile_ids)
        if offset != called_offset
    )
    return Meld(
        kind="chi",
        actor=actor,
        source=source,
        called_tile=called_tile,
        consumed_tiles=consumed_tiles,
        raw_value=value,
    )


def _decode_pon_or_kakan(actor: int, source: int, value: int) -> Meld:
    added = (value >> 5) & 0x3
    base_and_called = value >> 9
    tile_base = base_and_called // 3
    called_index = base_and_called % 3

    available = [tile_base * 4 + offset for offset in range(4)]
    tile_ids = [tile_id for offset, tile_id in enumerate(available) if offset != added]
    called_tile = tile136_to_string(tile_ids[called_index])

    if value & 0x8:
        consumed_tiles = tuple(
            tile136_to_string(tile_id)
            for idx, tile_id in enumerate(tile_ids)
            if idx != called_index
        )
        return Meld(
            kind="pon",
            actor=actor,
            source=source,
            called_tile=called_tile,
            consumed_tiles=consumed_tiles,
            raw_value=value,
        )

    added_tile = tile136_to_string(available[added])
    consumed_tiles = tuple(
        tile136_to_string(tile_id)
        for idx, tile_id in enumerate(tile_ids)
        if idx != called_index
    )
    return Meld(
        kind="kakan",
        actor=actor,
        source=source,
        called_tile=called_tile,
        consumed_tiles=consumed_tiles + (added_tile,),
        raw_value=value,
    )


def _decode_kan(actor: int, source: int, value: int) -> Meld:
    anchor_tile136 = value >> 8
    tile_base = anchor_tile136 // 4
    tile_ids = [tile_base * 4 + offset for offset in range(4)]
    tiles = tuple(tile136_to_string(tile_id) for tile_id in tile_ids)

    kind = "ankan" if actor == source else "daiminkan"
    if kind == "ankan":
        called_tile = tiles[0]
        consumed_tiles = tiles
    else:
        called_tile = tile136_to_string(anchor_tile136)
        consumed_tiles = tuple(
            tile136_to_string(tile_id)
            for tile_id in tile_ids
            if tile_id != anchor_tile136
        )

    return Meld(
        kind=kind,
        actor=actor,
        source=source,
        called_tile=called_tile,
        consumed_tiles=consumed_tiles,
        raw_value=value,
    )


def _decode_nuki(actor: int, source: int, value: int) -> Meld:
    tile = tile136_to_string(value >> 8)
    return Meld(
        kind="nuki",
        actor=actor,
        source=source,
        called_tile=tile,
        consumed_tiles=(),
        raw_value=value,
    )
