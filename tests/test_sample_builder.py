from __future__ import annotations

from pathlib import Path

from mjai.data.builders.sample_builder import TurnSampleBuilder
from mjai.data.tenhou.parser import TenhouParser


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_build_turn_samples_from_real_fixtures():
    parser = TenhouParser()
    builder = TurnSampleBuilder()

    total_samples = 0
    for path in sorted(FIXTURE_DIR.glob("*.xml")):
        samples = builder.build_samples(parser.parse_file(path))
        assert samples, f"expected turn samples from {path.name}"
        total_samples += len(samples)

    assert total_samples > 0


def test_turn_sample_label_is_in_legal_actions():
    parser = TenhouParser()
    builder = TurnSampleBuilder()
    path = FIXTURE_DIR / "2023010100gm-00e1-0000-8a80056a.xml"

    samples = builder.build_samples(parser.parse_file(path))

    for sample in samples:
        assert sample.label_action in sample.legal_actions


def test_turn_samples_include_riichi_discard_labels_on_real_fixture():
    parser = TenhouParser()
    builder = TurnSampleBuilder()

    riichi_count = 0
    for path in sorted(FIXTURE_DIR.glob("*.xml")):
        samples = builder.build_samples(parser.parse_file(path))
        riichi_count += sum(1 for sample in samples if sample.label_action["kind"] == "riichi_discard")

    assert riichi_count > 0


def test_riichi_discard_labels_are_present_in_legal_actions_on_real_fixtures():
    parser = TenhouParser()
    builder = TurnSampleBuilder()

    for path in sorted(FIXTURE_DIR.glob("*.xml")):
        samples = builder.build_samples(parser.parse_file(path))
        for sample in samples:
            if sample.label_action["kind"] == "riichi_discard":
                assert sample.label_action in sample.legal_actions


def test_turn_samples_include_tsumo_and_ankan_labels_on_real_fixtures():
    parser = TenhouParser()
    builder = TurnSampleBuilder()

    tsumo_count = 0
    ankan_count = 0
    for path in sorted(FIXTURE_DIR.glob("*.xml")):
        samples = builder.build_samples(parser.parse_file(path))
        tsumo_count += sum(1 for sample in samples if sample.label_action["kind"] == "tsumo")
        ankan_count += sum(1 for sample in samples if sample.label_action["kind"] == "ankan")

    assert tsumo_count > 0
    assert ankan_count > 0


def test_turn_samples_keep_tsumo_and_ankan_labels_in_legal_actions():
    parser = TenhouParser()
    builder = TurnSampleBuilder()

    found_tsumo = False
    found_ankan = False
    for path in sorted(FIXTURE_DIR.glob("*.xml")):
        samples = builder.build_samples(parser.parse_file(path))
        for sample in samples:
            if sample.label_action["kind"] == "tsumo":
                found_tsumo = True
                assert sample.label_action in sample.legal_actions
            if sample.label_action["kind"] == "ankan":
                found_ankan = True
                assert sample.label_action in sample.legal_actions

    assert found_tsumo
    assert found_ankan
