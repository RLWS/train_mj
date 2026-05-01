from __future__ import annotations

from dataclasses import dataclass

SUIT_ORDER = ("m", "p", "s", "z")
AKA_TILE_STRINGS = {"0m", "0p", "0s"}
AKA_BASE_INDICES = {4, 13, 22}
AKA_TILE136 = {16: "0m", 52: "0p", 88: "0s"}


@dataclass(frozen=True, slots=True)
class Tile:
    """Exact tile representation with Mahjong Soul red-five support."""

    text: str
    base_index: int
    is_aka: bool


def tile_to_base_index(tile: str) -> int:
    """Convert an exact tile string to a 34-tile base index."""
    if len(tile) != 2:
        raise ValueError(f"invalid tile format: {tile}")

    number_text, suit = tile
    if suit not in SUIT_ORDER:
        raise ValueError(f"invalid tile suit: {tile}")

    if suit == "z":
        if number_text not in "1234567":
            raise ValueError(f"invalid honor tile: {tile}")
        return 27 + int(number_text) - 1

    if number_text == "0":
        return {"m": 4, "p": 13, "s": 22}[suit]

    if number_text not in "123456789":
        raise ValueError(f"invalid suited tile: {tile}")

    suit_offset = {"m": 0, "p": 9, "s": 18}[suit]
    return suit_offset + int(number_text) - 1


def base_index_to_tile(index: int) -> str:
    """Convert a 34-tile base index to the non-red tile string."""
    if not 0 <= index < 34:
        raise ValueError(f"tile index out of range: {index}")

    if index < 9:
        return f"{index + 1}m"
    if index < 18:
        return f"{index - 8}p"
    if index < 27:
        return f"{index - 17}s"
    return f"{index - 26}z"


def is_aka_tile(tile: str) -> bool:
    return tile in AKA_TILE_STRINGS


def is_aka_index(index: int) -> bool:
    return index in AKA_BASE_INDICES


def parse_tile(tile: str) -> Tile:
    """Parse an exact tile string into an immutable tile object."""
    base_index = tile_to_base_index(tile)
    return Tile(text=tile, base_index=base_index, is_aka=is_aka_tile(tile))


def normalize_tile(tile: str) -> str:
    """Convert an exact tile string to the corresponding non-red tile string."""
    return base_index_to_tile(tile_to_base_index(tile))


def dora_from_indicator(indicator: str) -> str:
    """Return the non-red dora tile for a Tenhou dora indicator."""
    base_index = tile_to_base_index(indicator)
    if base_index < 27:
        suit_offset = (base_index // 9) * 9
        rank = base_index % 9
        return base_index_to_tile(suit_offset + ((rank + 1) % 9))

    honor_index = base_index - 27
    if honor_index < 4:
        return base_index_to_tile(27 + ((honor_index + 1) % 4))
    return base_index_to_tile(31 + ((honor_index - 4 + 1) % 3))


def tile136_to_base_index(tile136: int) -> int:
    """Convert a Tenhou 136-tile identifier to a 34-tile base index."""
    if not 0 <= tile136 < 136:
        raise ValueError(f"tile136 out of range: {tile136}")
    return tile136 // 4


def is_aka_tile136(tile136: int) -> bool:
    """Return whether a Tenhou 136-tile identifier is a red five."""
    return tile136 in AKA_TILE136


def tile136_to_string(tile136: int) -> str:
    """Convert a Tenhou 136-tile identifier to an exact tile string."""
    if tile136 in AKA_TILE136:
        return AKA_TILE136[tile136]
    return base_index_to_tile(tile136_to_base_index(tile136))
