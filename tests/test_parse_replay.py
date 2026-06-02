from __future__ import annotations

from pathlib import Path

from mjai.data.tenhou.parser import TenhouParser


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def _fixture_paths() -> list[Path]:
    """返回测试用天凤 XML 夹具文件路径列表。
    
    参数:
        无。
    
    返回:
        list[Path]: 夹具 XML 文件路径列表。
    
    异常:
        无。
    """
    return sorted(FIXTURE_DIR.glob("*.xml"))


def test_parse_all_real_fixtures():
    """验证全部真实天凤夹具都可被解析。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    parser = TenhouParser()
    paths = _fixture_paths()

    assert paths, "expected at least one Tenhou XML fixture"

    for path in paths:
        events = parser.parse_file(path)
        assert events, f"no events parsed from {path.name}"
        assert events[0].event_type == "go"
        assert any(event.event_type == "init" for event in events)
        assert events[-1].event_type in {"agari", "ryuukyoku"}


def test_parse_real_fixture_payload_shapes():
    """验证真实夹具解析后的载荷结构。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    parser = TenhouParser()
    path = _fixture_paths()[0]
    events = parser.parse_file(path)

    init_event = next(event for event in events if event.event_type == "init")
    first_draw = next(event for event in events if event.event_type == "draw")
    first_discard = next(event for event in events if event.event_type == "discard")

    assert init_event.payload["oya"] in {0, 1, 2, 3}
    assert init_event.payload["round_wind"] in {0, 1, 2, 3}
    assert isinstance(init_event.payload["dora_indicator"], str)
    assert len(init_event.payload["hai0"]) == 13
    assert len(init_event.payload["hai1"]) == 13
    assert len(init_event.payload["hai2"]) == 13
    assert len(init_event.payload["hai3"]) == 13
    assert isinstance(first_draw.payload["tile"], str)
    assert isinstance(first_discard.payload["tile"], str)
    assert isinstance(first_discard.payload["tsumogiri"], bool)
