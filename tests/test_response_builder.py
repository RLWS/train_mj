from __future__ import annotations

from pathlib import Path

from mjai.data.builders.sample_builder import ResponseSampleBuilder
from mjai.data.schema.events import Event
from mjai.data.tenhou.parser import TenhouParser


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_build_response_samples_from_real_fixtures():
    parser = TenhouParser()
    builder = ResponseSampleBuilder()

    total_samples = 0
    for path in sorted(FIXTURE_DIR.glob("*.xml")):
        samples = builder.build_samples(parser.parse_file(path))
        total_samples += len(samples)

    assert total_samples > 0


def test_response_sample_label_is_in_legal_actions():
    parser = TenhouParser()
    builder = ResponseSampleBuilder()
    path = FIXTURE_DIR / "2023010100gm-00e1-0000-8a80056a.xml"

    samples = builder.build_samples(parser.parse_file(path))

    for sample in samples:
        assert sample.label_action in sample.legal_actions


def test_response_sample_label_after_open_meld_clears_temporary_furiten():
    parser = TenhouParser()
    builder = ResponseSampleBuilder()
    path = FIXTURE_DIR / "2023011001gm-00a9-0000-ef310a72.xml"

    samples = builder.build_samples(parser.parse_file(path))
    target_sample = next(
        sample
        for sample in samples
        if sample.sample_id == "2023011001gm-00a9-0000-ef310a72:8:102:3:response"
    )

    assert target_sample.state["temporary_furiten"] is False
    assert target_sample.label_action["kind"] == "ron"
    assert target_sample.label_action in target_sample.legal_actions


def test_response_samples_include_non_pass_labels():
    parser = TenhouParser()
    builder = ResponseSampleBuilder()
    path = FIXTURE_DIR / "2023010100gm-00e1-0000-8a80056a.xml"

    samples = builder.build_samples(parser.parse_file(path))
    non_pass = [sample for sample in samples if sample.label_action["kind"] != "pass"]

    assert non_pass


def test_response_samples_include_ron_labels_on_real_fixtures():
    parser = TenhouParser()
    builder = ResponseSampleBuilder()

    ron_count = 0
    for path in sorted(FIXTURE_DIR.glob("*.xml")):
        samples = builder.build_samples(parser.parse_file(path))
        ron_count += sum(1 for sample in samples if sample.label_action["kind"] == "ron")

    assert ron_count > 0


def test_response_sample_furiten_omits_ron_from_legal_actions():
    builder = ResponseSampleBuilder()
    events = [
        Event(
            game_id="furiten-case",
            kyoku_index=0,
            step_index=0,
            event_type="init",
            payload={
                "hai0": ["1z"] * 12 + ["5z"],
                "hai1": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"],
                "hai2": ["2z"] * 12 + ["5z"],
                "hai3": ["3z"] * 13,
                "ten": "250,250,250,250",
            },
        ),
        Event(
            game_id="furiten-case",
            kyoku_index=0,
            step_index=1,
            event_type="draw",
            actor=1,
            payload={"tile": "9s"},
        ),
        Event(
            game_id="furiten-case",
            kyoku_index=0,
            step_index=2,
            event_type="discard",
            actor=1,
            payload={"tile": "5z", "tsumogiri": False},
        ),
        Event(
            game_id="furiten-case",
            kyoku_index=0,
            step_index=3,
            event_type="draw",
            actor=0,
            payload={"tile": "4z"},
        ),
        Event(
            game_id="furiten-case",
            kyoku_index=0,
            step_index=4,
            event_type="discard",
            actor=0,
            payload={"tile": "5z", "tsumogiri": False},
        ),
        Event(
            game_id="furiten-case",
            kyoku_index=0,
            step_index=5,
            event_type="draw",
            actor=2,
            payload={"tile": "4z"},
        ),
    ]

    samples = builder.build_samples(events)
    target_sample = next(sample for sample in samples if sample.sample_id == "furiten-case:0:4:1:response")

    assert target_sample.label_action["kind"] == "pass"
    assert all(action["kind"] != "ron" for action in target_sample.legal_actions)


def test_response_samples_mark_temporary_furiten_after_passing_ron():
    builder = ResponseSampleBuilder()
    events = [
        Event(
            game_id="temp-furiten-case",
            kyoku_index=0,
            step_index=0,
            event_type="init",
            payload={
                "hai0": ["1z"] * 12 + ["5z"],
                "hai1": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"],
                "hai2": ["2z"] * 12 + ["5z"],
                "hai3": ["3z"] * 13,
                "ten": "250,250,250,250",
            },
        ),
        Event(
            game_id="temp-furiten-case",
            kyoku_index=0,
            step_index=1,
            event_type="draw",
            actor=0,
            payload={"tile": "4z"},
        ),
        Event(
            game_id="temp-furiten-case",
            kyoku_index=0,
            step_index=2,
            event_type="discard",
            actor=0,
            payload={"tile": "5z", "tsumogiri": False},
        ),
        Event(
            game_id="temp-furiten-case",
            kyoku_index=0,
            step_index=3,
            event_type="draw",
            actor=2,
            payload={"tile": "4z"},
        ),
        Event(
            game_id="temp-furiten-case",
            kyoku_index=0,
            step_index=4,
            event_type="discard",
            actor=2,
            payload={"tile": "5z", "tsumogiri": False},
        ),
        Event(
            game_id="temp-furiten-case",
            kyoku_index=0,
            step_index=5,
            event_type="draw",
            actor=1,
            payload={"tile": "9s"},
        ),
    ]

    samples = builder.build_samples(events)
    first_response = next(sample for sample in samples if sample.sample_id == "temp-furiten-case:0:2:1:response")
    second_response = next(sample for sample in samples if sample.sample_id == "temp-furiten-case:0:4:1:response")

    assert first_response.label_action["kind"] == "pass"
    assert any(action["kind"] == "ron" for action in first_response.legal_actions)
    assert first_response.state["temporary_furiten"] is False
    assert second_response.state["temporary_furiten"] is True
    assert all(action["kind"] != "ron" for action in second_response.legal_actions)


def test_response_samples_clear_temporary_furiten_after_self_draw():
    builder = ResponseSampleBuilder()
    events = [
        Event(
            game_id="temp-furiten-clear",
            kyoku_index=0,
            step_index=0,
            event_type="init",
            payload={
                "hai0": ["1z"] * 12 + ["5z"],
                "hai1": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"],
                "hai2": ["2z"] * 12 + ["5z"],
                "hai3": ["3z"] * 13,
                "ten": "250,250,250,250",
            },
        ),
        Event(
            game_id="temp-furiten-clear",
            kyoku_index=0,
            step_index=1,
            event_type="draw",
            actor=0,
            payload={"tile": "4z"},
        ),
        Event(
            game_id="temp-furiten-clear",
            kyoku_index=0,
            step_index=2,
            event_type="discard",
            actor=0,
            payload={"tile": "5z", "tsumogiri": False},
        ),
        Event(
            game_id="temp-furiten-clear",
            kyoku_index=0,
            step_index=3,
            event_type="draw",
            actor=1,
            payload={"tile": "9s"},
        ),
        Event(
            game_id="temp-furiten-clear",
            kyoku_index=0,
            step_index=4,
            event_type="discard",
            actor=1,
            payload={"tile": "9s", "tsumogiri": True},
        ),
        Event(
            game_id="temp-furiten-clear",
            kyoku_index=0,
            step_index=5,
            event_type="draw",
            actor=2,
            payload={"tile": "4z"},
        ),
        Event(
            game_id="temp-furiten-clear",
            kyoku_index=0,
            step_index=6,
            event_type="discard",
            actor=2,
            payload={"tile": "5z", "tsumogiri": False},
        ),
        Event(
            game_id="temp-furiten-clear",
            kyoku_index=0,
            step_index=7,
            event_type="draw",
            actor=3,
            payload={"tile": "4z"},
        ),
    ]

    samples = builder.build_samples(events)
    second_response = next(sample for sample in samples if sample.sample_id == "temp-furiten-clear:0:6:1:response")

    assert second_response.state["temporary_furiten"] is False
    assert any(action["kind"] == "ron" for action in second_response.legal_actions)


def test_response_samples_mark_riichi_furiten_after_passing_ron():
    builder = ResponseSampleBuilder()
    events = [
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=0,
            event_type="init",
            payload={
                "hai0": ["1z"] * 12 + ["5z"],
                "hai1": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"],
                "hai2": ["2z"] * 12 + ["5z"],
                "hai3": ["3z"] * 13,
                "ten": "250,250,250,250",
            },
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=1,
            event_type="draw",
            actor=1,
            payload={"tile": "9s"},
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=2,
            event_type="reach",
            actor=1,
            payload={"step": 1},
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=3,
            event_type="discard",
            actor=1,
            payload={"tile": "9s", "tsumogiri": True},
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=4,
            event_type="reach",
            actor=1,
            payload={"step": 2},
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=5,
            event_type="draw",
            actor=0,
            payload={"tile": "4z"},
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=6,
            event_type="discard",
            actor=0,
            payload={"tile": "5z", "tsumogiri": False},
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=7,
            event_type="draw",
            actor=2,
            payload={"tile": "4z"},
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=8,
            event_type="discard",
            actor=2,
            payload={"tile": "5z", "tsumogiri": False},
        ),
        Event(
            game_id="riichi-furiten-case",
            kyoku_index=0,
            step_index=9,
            event_type="draw",
            actor=3,
            payload={"tile": "4z"},
        ),
    ]

    samples = builder.build_samples(events)
    first_response = next(sample for sample in samples if sample.sample_id == "riichi-furiten-case:0:6:1:response")
    second_response = next(sample for sample in samples if sample.sample_id == "riichi-furiten-case:0:8:1:response")

    assert first_response.label_action["kind"] == "pass"
    assert any(action["kind"] == "ron" for action in first_response.legal_actions)
    assert second_response.state["riichi_furiten"] is True
    assert all(action["kind"] != "ron" for action in second_response.legal_actions)


def test_response_samples_include_chankan_on_kakan_window():
    builder = ResponseSampleBuilder()
    events = [
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=0,
            event_type="init",
            payload={
                "hai0": ["2m", "3m", "4m", "6m", "7m", "1p", "2p", "3p", "4s", "4s", "5m", "5m", "5m"],
                "hai1": ["1m", "2m", "3m", "4m", "6m", "7m", "8m", "2p", "3p", "4p", "5z", "5z", "9s"],
                "hai2": ["2z"] * 12 + ["5m"],
                "hai3": ["3z"] * 13,
                "ten": "250,250,250,250",
            },
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=1,
            event_type="discard",
            actor=2,
            payload={"tile": "5m", "tsumogiri": False},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=2,
            event_type="meld",
            actor=0,
            payload={
                "kind": "pon",
                "source": 2,
                "called_tile": "5m",
                "consumed_tiles": ["5m", "5m"],
                "raw_value": 0,
            },
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=3,
            event_type="discard",
            actor=0,
            payload={"tile": "1p", "tsumogiri": False},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=4,
            event_type="draw",
            actor=1,
            payload={"tile": "9m"},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=5,
            event_type="discard",
            actor=1,
            payload={"tile": "9s", "tsumogiri": False},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=6,
            event_type="draw",
            actor=2,
            payload={"tile": "1z"},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=7,
            event_type="discard",
            actor=2,
            payload={"tile": "1z", "tsumogiri": True},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=8,
            event_type="draw",
            actor=3,
            payload={"tile": "1z"},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=9,
            event_type="discard",
            actor=3,
            payload={"tile": "1z", "tsumogiri": True},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=10,
            event_type="draw",
            actor=0,
            payload={"tile": "2z"},
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=11,
            event_type="meld",
            actor=0,
            payload={
                "kind": "kakan",
                "source": 2,
                "called_tile": "5m",
                "consumed_tiles": ["5m", "5m", "5m"],
                "raw_value": 0,
            },
        ),
        Event(
            game_id="chankan-case",
            kyoku_index=0,
            step_index=12,
            event_type="agari",
            actor=1,
            payload={"who": "1", "fromWho": "0"},
        ),
    ]

    samples = builder.build_samples(events)
    target_sample = next(sample for sample in samples if sample.sample_id == "chankan-case:0:11:1:response")

    assert target_sample.label_action["kind"] == "chankan"
    assert any(action["kind"] == "chankan" for action in target_sample.legal_actions)


def test_response_samples_skip_applying_kakan_when_chankan_resolves():
    builder = ResponseSampleBuilder()
    events = [
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=0,
            event_type="init",
            payload={
                "hai0": ["2m", "3m", "4m", "6m", "7m", "1p", "2p", "3p", "4s", "4s", "5m", "5m", "5m"],
                "hai1": ["1m", "2m", "3m", "4m", "6m", "7m", "8m", "2p", "3p", "4p", "5z", "5z", "9s"],
                "hai2": ["2z"] * 12 + ["5m"],
                "hai3": ["3z"] * 13,
                "ten": "250,250,250,250",
            },
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=1,
            event_type="discard",
            actor=2,
            payload={"tile": "5m", "tsumogiri": False},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=2,
            event_type="meld",
            actor=0,
            payload={
                "kind": "pon",
                "source": 2,
                "called_tile": "5m",
                "consumed_tiles": ["5m", "5m"],
                "raw_value": 0,
            },
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=3,
            event_type="discard",
            actor=0,
            payload={"tile": "1p", "tsumogiri": False},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=4,
            event_type="draw",
            actor=1,
            payload={"tile": "9m"},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=5,
            event_type="discard",
            actor=1,
            payload={"tile": "9s", "tsumogiri": False},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=6,
            event_type="draw",
            actor=2,
            payload={"tile": "1z"},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=7,
            event_type="discard",
            actor=2,
            payload={"tile": "1z", "tsumogiri": True},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=8,
            event_type="draw",
            actor=3,
            payload={"tile": "1z"},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=9,
            event_type="discard",
            actor=3,
            payload={"tile": "1z", "tsumogiri": True},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=10,
            event_type="draw",
            actor=0,
            payload={"tile": "2z"},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=11,
            event_type="meld",
            actor=0,
            payload={
                "kind": "kakan",
                "source": 2,
                "called_tile": "5m",
                "consumed_tiles": ["5m", "5m", "5m"],
                "raw_value": 0,
            },
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=12,
            event_type="agari",
            actor=1,
            payload={"who": "1", "fromWho": "0"},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=13,
            event_type="draw",
            actor=0,
            payload={"tile": "1z"},
        ),
        Event(
            game_id="chankan-state-case",
            kyoku_index=0,
            step_index=14,
            event_type="discard",
            actor=0,
            payload={"tile": "5m", "tsumogiri": False},
        ),
    ]

    samples = builder.build_samples(events)
    post_agari_sample = next(sample for sample in samples if sample.sample_id == "chankan-state-case:0:14:1:response")

    assert post_agari_sample.state["response_target"]["tile"] == "5m"
