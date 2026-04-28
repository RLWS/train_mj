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
                "hai2": ["2z"] * 13,
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
