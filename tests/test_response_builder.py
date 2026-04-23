from __future__ import annotations

from pathlib import Path

from mjai.data.builders.sample_builder import ResponseSampleBuilder
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
