from __future__ import annotations

from mjai.rules.legal_actions import legal_discards, legal_responses, legal_turn_actions
from mjai.rules.state import GameState, HandState, MeldState


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


def test_legal_responses_include_ron_when_hand_is_complete():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"]
    )

    actions = legal_responses(hand=hand, actor=1, source=0, tile="5z")
    normalized = {(action.kind, action.tile) for action in actions}

    assert ("ron", "5z") in normalized


def test_legal_responses_exclude_ron_when_actor_is_furiten():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"]
    )
    state = GameState(hands=[HandState(), hand, HandState(), HandState()])
    state.furiten_discard_bases[1].add(31)

    actions = legal_responses(hand=hand, actor=1, source=0, tile="5z", state=state)
    normalized = {(action.kind, action.tile) for action in actions}

    assert ("ron", "5z") not in normalized


def test_legal_responses_exclude_ron_when_actor_is_temporarily_furiten():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"]
    )
    state = GameState(hands=[HandState(), hand, HandState(), HandState()])
    state.temporary_furiten[1] = True

    actions = legal_responses(hand=hand, actor=1, source=0, tile="5z", state=state)
    normalized = {(action.kind, action.tile) for action in actions}

    assert ("ron", "5z") not in normalized


def test_legal_responses_exclude_ron_when_actor_is_riichi_furiten():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"]
    )
    state = GameState(hands=[HandState(), hand, HandState(), HandState()])
    state.riichi_furiten[1] = True

    actions = legal_responses(hand=hand, actor=1, source=0, tile="5z", state=state)
    normalized = {(action.kind, action.tile) for action in actions}

    assert ("ron", "5z") not in normalized


def test_legal_responses_include_chankan_when_enabled():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"]
    )

    actions = legal_responses(hand=hand, actor=1, source=0, tile="5z", allow_chankan=True)
    normalized = {(action.kind, action.tile) for action in actions}

    assert ("chankan", "5z") in normalized


def test_legal_responses_chankan_window_excludes_call_actions():
    hand = HandState.from_tiles(["3m", "4m", "6m", "7m", "5z"] * 2 + ["1p", "2p", "3p"])

    actions = legal_responses(hand=hand, actor=1, source=0, tile="5z", allow_chankan=True)
    kinds = {action.kind for action in actions}

    assert "pass" in kinds
    assert "chi" not in kinds
    assert "pon" not in kinds
    assert "daiminkan" not in kinds


def test_legal_turn_actions_include_tsumo_for_complete_hand():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z", "5z"]
    )

    actions = legal_turn_actions(hand)
    normalized = {(action.kind, action.tile) for action in actions}

    assert ("tsumo", None) in normalized


def test_legal_turn_actions_include_ankan_when_four_of_a_kind_exists():
    hand = HandState.from_tiles(["1p", "1p", "1p", "1p", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1z", "1z"])

    actions = legal_turn_actions(hand, state=GameState(), actor=0)
    ankan_actions = [action for action in actions if action.kind == "ankan"]

    assert ankan_actions
    assert ankan_actions[0].consumed_tiles == ("1p", "1p", "1p", "1p")


def test_legal_turn_actions_include_kakan_when_matching_pon_exists():
    hand = HandState.from_tiles(["5m", "2m", "3m", "4m", "6m", "7m", "8m", "1p", "2p", "3p", "4s"])
    state = GameState(
        hands=[hand, HandState(), HandState(), HandState()],
        melds=[
            [MeldState(kind="pon", source=1, called_tile="5m", consumed_tiles=("0m", "5m"))],
            [],
            [],
            [],
        ],
    )

    actions = legal_turn_actions(hand, state=state, actor=0)
    normalized = {(action.kind, action.tile, action.consumed_tiles, action.source) for action in actions}

    assert ("kakan", "5m", ("0m", "5m", "5m"), 1) in normalized


def test_legal_turn_actions_include_kyuushu_kyuuhai_on_first_draw_with_nine_terminals():
    hand = HandState.from_tiles(
        ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z", "5m"]
    )
    state = GameState(hands=[hand, HandState(), HandState(), HandState()])

    actions = legal_turn_actions(hand, state=state, actor=0)
    normalized = {(action.kind, action.tile) for action in actions}

    assert ("kyuushu_kyuuhai", None) in normalized


def test_legal_turn_actions_exclude_kyuushu_kyuuhai_after_actor_has_discarded():
    hand = HandState.from_tiles(
        ["1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "4z", "5z", "6z", "7z", "5m"]
    )
    state = GameState(hands=[hand, HandState(), HandState(), HandState()])
    state.discards[0].append("5m")

    actions = legal_turn_actions(hand, state=state, actor=0)
    assert all(action.kind != "kyuushu_kyuuhai" for action in actions)
