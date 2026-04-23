from __future__ import annotations

from pathlib import Path

from mjai.data.tenhou.parser import TenhouParser
from mjai.rules.validators import ReplayValidator


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_validate_all_real_fixtures():
    parser = TenhouParser()
    validator = ReplayValidator()

    for path in sorted(FIXTURE_DIR.glob("*.xml")):
        events = parser.parse_file(path)
        summary = validator.validate_events(events)

        assert summary.game_id == path.stem
        assert summary.hands_seen >= 1
        assert summary.total_events == len(events)
        assert summary.total_discards > 0


def test_validate_fixture_with_melds_reports_nonzero_melds():
    parser = TenhouParser()
    validator = ReplayValidator()
    path = FIXTURE_DIR / "2023010100gm-00e1-0000-8a80056a.xml"

    summary = validator.validate_events(parser.parse_file(path))

    assert summary.total_melds > 0
