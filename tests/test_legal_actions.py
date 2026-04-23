from __future__ import annotations

from mjai.rules.legal_actions import legal_discards, legal_responses
from mjai.rules.state import HandState


def test_legal_discards_include_red_and_normal_five_when_both_exist():
    hand = HandState.from_tiles(["0m", "5m", "6m", "1z"])

    discards = legal_discards(hand)
    tiles = [action.tile for action in discards]

    assert "0m" in tiles
    assert "5m" in tiles
    assert "6m" in tiles
    assert "1z" in tiles


def test_legal_discards_do_not_include_missing_tiles():
    hand = HandState.from_tiles(["1m", "1m", "3p"])

    discards = legal_discards(hand)
    tiles = {action.tile for action in discards}

    assert "1m" in tiles
    assert "3p" in tiles
    assert "2m" not in tiles
    assert "0m" not in tiles


def test_legal_responses_include_pass_pon_and_daiminkan():
    hand = HandState.from_tiles(["0m", "5m", "5m", "2p"])

    actions = legal_responses(hand=hand, actor=1, source=0, tile="5m")
    normalized = {(action.kind, action.tile, action.consumed_tiles) for action in actions}

    assert ("pass", None, ()) in normalized
    assert ("pon", "5m", ("0m", "5m")) in normalized
    assert ("pon", "5m", ("5m", "5m")) in normalized
    assert ("daiminkan", "5m", ("0m", "5m", "5m")) in normalized


def test_legal_responses_include_chi_only_from_upper_player():
    hand = HandState.from_tiles(["3m", "0m", "6m", "7m"])

    upper_actions = legal_responses(hand=hand, actor=0, source=3, tile="4m")
    other_actions = legal_responses(hand=hand, actor=0, source=1, tile="4m")

    upper_chi = {(action.kind, action.consumed_tiles) for action in upper_actions if action.kind == "chi"}
    other_chi = {(action.kind, action.consumed_tiles) for action in other_actions if action.kind == "chi"}

    assert ("chi", ("3m", "0m")) in upper_chi
    assert ("chi", ("0m", "6m")) in upper_chi
    assert other_chi == set()
