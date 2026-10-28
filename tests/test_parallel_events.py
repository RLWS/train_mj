from pathlib import Path
import json

import pyarrow.parquet as pq
import pytest

from mjai.cli import main
from mjai.data.builders.event_builder import EventBuilder
from mjai.data.builders.event_validator import EventValidator
from mjai.data.datasets.event_store import write_events_to_parquet
from mjai.utils.process_workers import process_map

FIXTURES = Path("tests/fixtures/tenhou")


def test_parallel_preprocess_matches_serial(tmp_path):
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    expected = EventBuilder().build_from_xml_dir(FIXTURES, serial)
    assert main(["preprocess", "--input", str(FIXTURES), "--output", str(parallel),
                 "--workers", "2"]) == 0
    assert json.loads((serial / "preprocess_summary.json").read_text()) == json.loads(
        (parallel / "preprocess_summary.json").read_text())
    assert len(list(parallel.glob("*.parquet"))) == expected.games
    for path in serial.glob("*.parquet"):
        assert pq.read_table(path).equals(pq.read_table(parallel / path.name))


def test_parallel_validation_matches_serial_including_failure(tmp_path):
    bronze = tmp_path / "bronze"
    EventBuilder().build_from_xml_dir(FIXTURES, bronze)
    write_events_to_parquet([], bronze / "invalid.parquet")
    serial = tmp_path / "serial.json"
    parallel = tmp_path / "parallel.json"
    result = EventValidator().validate_dir(bronze, serial)
    assert result.files_failed == 1
    assert main(["validate-events", "--input", str(bronze), "--report", str(parallel),
                 "--workers", "2"]) == 0
    assert json.loads(serial.read_text()) == json.loads(parallel.read_text())


def test_parallel_empty_inputs(tmp_path):
    assert EventBuilder().build_from_xml_dir(tmp_path, tmp_path / "out", workers=2).games == 0
    result = EventValidator().validate_dir(tmp_path, tmp_path / "report.json", workers=2)
    assert result.files_seen == 0
    assert result.warnings


def test_process_map_propagates_worker_error():
    with pytest.raises(ValueError):
        list(process_map(int, ["1", "invalid"], workers=2))
