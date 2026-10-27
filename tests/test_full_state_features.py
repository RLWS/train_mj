from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from mjai.data.builders.sample_builder import ResponseSampleBuilder, TurnSampleBuilder, _player_view_snapshot
from mjai.data.builders.dataset_builder import _build_one_game
from mjai.data.datasets.sample_store import serialize_sample
from mjai.data.tenhou.parser import TenhouParser
from mjai.features.action_features import encode_response_candidate_actions, encode_turn_candidate_actions
from mjai.features.state_features import encode_response_state, encode_turn_state
from mjai.rules.state import GameState, HandState
from mjai.serving.checkpoint import load_policy_checkpoint
from mjai.serving.validation import RequestValidationError, validate_and_normalize_state
from tests.test_serving_predictor import _response_state, _write_checkpoint


FIXTURES = Path("tests/fixtures/tenhou")


def test_global_seat_rotation_preserves_state_and_candidate_encodings():
    state = _response_state()
    state["scores"] = [21000, 33000, 16000, 30000]
    state["all_melds"][2] = [{"kind": "pon", "source": 0, "called_tile": "5z", "consumed_tiles": ["5z", "5z"]}]
    state["discard_history"] = [{"actor": 0, "tile": "3m", "tsumogiri": True, "riichi_discard": False}]
    state["public_events"] = [{"kind": "pon", "actor": 2, "source": 0, "tile": "5z", "discard_count": 1}]
    actions = [{"kind": "pass"}, {"kind": "chi", "source": 0, "tile": "3m", "consumed_tiles": ["1m", "2m"]}]
    original = deepcopy(state)
    for rotation in (1, 2, 3):
        rotated = deepcopy(state)
        for key in ("actor", "dealer", "last_discard_actor"):
            rotated[key] = (state[key] + rotation) % 4
        for key in ("all_discards", "all_melds", "scores", "riichi_declared_by_player", "riichi_turn_by_player"):
            rotated[key] = [deepcopy(state[key][(i - rotation) % 4]) for i in range(4)]
        for row in rotated["all_melds"]:
            for meld in row:
                meld["source"] = (meld["source"] + rotation) % 4
        for key in ("discard_history", "public_events"):
            for item in rotated[key]:
                for seat in ("actor", "source"):
                    if seat in item:
                        item[seat] = (item[seat] + rotation) % 4
        rotated["response_target"]["source"] = rotation
        rotated_actions = deepcopy(actions)
        rotated_actions[1]["source"] = rotation
        assert encode_response_state(rotated) == encode_response_state(state)
        assert encode_response_candidate_actions(rotated_actions, rotated) == encode_response_candidate_actions(actions, state)
        turn_actions = [{"kind": "discard", "tile": "1m"}]
        assert encode_turn_candidate_actions(turn_actions, rotated) == encode_turn_candidate_actions(turn_actions, state)
    assert state == original


def test_meld_identity_round_and_wall_information_are_distinguishable():
    state = _response_state()
    baseline = encode_turn_state(state)
    for key, value in (("round_wind", 1), ("round_number", 3), ("match_rounds", 4), ("remaining_tiles", 3)):
        changed = deepcopy(state)
        changed[key] = value
        assert encode_turn_state(changed) != baseline
    changed = deepcopy(state)
    changed["all_melds"][2] = [{"kind": "pon", "source": 3, "called_tile": "0p", "consumed_tiles": ["5p", "5p"]}]
    red = encode_turn_state(changed)
    changed["all_melds"][2][0]["called_tile"] = "5p"
    assert baseline != red != encode_turn_state(changed)


def test_parser_uses_physical_draw_identity_for_uppercase_tsumogiri(tmp_path):
    path = tmp_path / "parser.xml"
    path.write_text('<mjloggm><GO type="9"/><INIT seed="7,2,1,0,0,4"/>'
                    '<T16/><D16/><U20/><E21/></mjloggm>')
    events = TenhouParser().parse_file(path)
    init = next(e for e in events if e.event_type == "init")
    assert (init.payload["round_wind"], init.payload["round_number"], init.payload["match_rounds"]) == (1, 3, 8)
    assert [e.payload["tsumogiri"] for e in events if e.event_type == "discard"] == [True, False]


def test_wall_counter_scores_and_public_history_do_not_leak_hidden_draws():
    state = GameState(hands=[HandState.from_tiles(["1m"]), HandState(), HandState(), HandState()])
    state.apply_draw(1, "9p")
    snapshot = _player_view_snapshot(state, 0)
    assert snapshot["remaining_tiles"] == 69
    assert snapshot["scores"] == [25000] * 4
    assert snapshot["public_events"] == []
    other = GameState(hands=[HandState.from_tiles(["1m"]), HandState(), HandState(), HandState()])
    other.apply_draw(1, "1s")
    assert encode_turn_state(snapshot) == encode_turn_state(_player_view_snapshot(other, 0))
    state.apply_reach(0, 1)
    state.apply_discard(0, "1m", False)
    state.apply_reach(0, 2)
    assert state.riichi_sticks == 1
    assert state.scores[0] == 240
    assert state.discard_details[0]["riichi_discard"] is True
    assert state.public_events[0]["discard_count"] == 0
    assert state.public_events[1]["discard_count"] == 1


def test_real_samples_match_serving_encoding_and_no_current_reach_leak():
    parser = TenhouParser()
    reached = False
    for path in sorted(FIXTURES.glob("*.xml")):
        events = parser.parse_file(path)
        for builder in (TurnSampleBuilder(), ResponseSampleBuilder()):
            samples = builder.build_samples(events)
            # Include calls, post-call turns and riichi decisions, not just initial hands.
            selected = [s for s in samples if s.state["own_melds"] or s.label_action["kind"] == "riichi_discard"]
            selected = samples[:2] + selected[:6]
            for sample in selected:
                normalized = validate_and_normalize_state(sample.decision_type, sample.state, require_full_state=True)
                state_encoder = encode_turn_state if sample.decision_type == "turn" else encode_response_state
                action_encoder = encode_turn_candidate_actions if sample.decision_type == "turn" else encode_response_candidate_actions
                assert state_encoder(normalized) == state_encoder(sample.state)
                assert action_encoder(sample.legal_actions, normalized) == action_encoder(sample.legal_actions, sample.state)
                serialize_sample(sample)
            for sample in samples:
                if sample.label_action["kind"] == "riichi_discard":
                    reached = True
                    assert not any(e.get("actor") == sample.actor and e["kind"] == "reach1" for e in sample.state["public_events"])
    assert reached


def test_reject_old_bronze_and_checkpoint(tmp_path):
    events = TenhouParser().parse_file(next(FIXTURES.glob("*.xml")))
    next(e for e in events if e.event_type == "init").payload.pop("feature_schema")
    with pytest.raises(ValueError, match="rerun preprocess"):
        _build_one_game("old", events, tmp_path, validate=False)
    path = _write_checkpoint(tmp_path / "old.pt", task="turn", hidden_size=8)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint.pop("feature_schema")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="rebuild data and retrain"):
        load_policy_checkpoint(path)


@pytest.mark.parametrize("field", ["all_melds", "remaining_tiles", "public_events", "feature_schema", "round_number"])
def test_serving_rejects_missing_full_state(field):
    state = _response_state()
    state.pop(field)
    with pytest.raises(RequestValidationError, match=field):
        validate_and_normalize_state("response", state, require_full_state=True)
