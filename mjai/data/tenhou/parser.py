from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import iterparse

from mjai.data.schema.events import Event
from mjai.data.tenhou.meld_decoder import decode_meld
from mjai.rules.tiles import tile136_to_string

DRAW_TAG_TO_ACTOR = {"T": 0, "U": 1, "V": 2, "W": 3}
DISCARD_TAG_TO_ACTOR = {"D": 0, "E": 1, "F": 2, "G": 3}


class TenhouParser:
    """Streaming parser for Tenhou XML logs."""

    def parse_file(self, path: str | Path) -> list[Event]:
        """解析单个天凤 XML 日志文件并生成规范化事件。
        
        参数:
            self (Self): 当前对象实例。
            path (str | Path): 文件或目录路径。
        
        返回:
            list[Event]: 列表结果。
        
        异常:
            无。
        """
        file_path = Path(path)
        game_id = file_path.stem
        kyoku_index = -1
        step_index = 0
        events: list[Event] = []
        last_draws: list[int | None] = [None] * 4
        game_type: int | None = None

        for _, elem in iterparse(file_path, events=("end",)):
            tag = elem.tag
            tag_head = tag[0] if tag else ""
            normalized_head = tag_head.upper()

            if tag == "GO":
                game_type = int(elem.get("type", "0"))
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="go",
                        actor=None,
                        raw_tag=tag,
                        payload={
                            "type": int(elem.get("type", "0")),
                            "lobby": int(elem.get("lobby", "0")),
                        },
                    )
                )
                step_index += 1
            elif tag == "UN":
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="un",
                        actor=None,
                        raw_tag=tag,
                        payload={
                            "n0": elem.get("n0", ""),
                            "n1": elem.get("n1", ""),
                            "n2": elem.get("n2", ""),
                            "n3": elem.get("n3", ""),
                        },
                    )
                )
                step_index += 1
            elif tag == "TAIKYOKU":
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="taikyoku",
                        actor=None,
                        raw_tag=tag,
                        payload={"oya": int(elem.get("oya", "0"))},
                    )
                )
                step_index += 1
            elif tag == "INIT":
                kyoku_index += 1
                step_index = 0
                seed = elem.get("seed")
                last_draws = [None] * 4
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="init",
                        actor=None,
                        raw_tag=tag,
                        payload={
                            "seed": seed,
                            "round_wind": _parse_round_wind(seed),
                            "round_number": int(seed.split(",")[0]) % 4 if seed else 0,
                            "match_rounds": None if game_type is None else (8 if game_type & 8 else 4),
                            "feature_schema": "full_state_v2",
                            "ten": elem.get("ten"),
                            "oya": int(elem.get("oya", "0")),
                            "dora_indicator": tile136_to_string(_parse_seed_dora_indicator(seed)),
                            "hai0": _parse_tiles_attr(elem.get("hai0")),
                            "hai1": _parse_tiles_attr(elem.get("hai1")),
                            "hai2": _parse_tiles_attr(elem.get("hai2")),
                            "hai3": _parse_tiles_attr(elem.get("hai3")),
                        },
                    )
                )
                step_index += 1
            elif tag == "DORA":
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="dora",
                        actor=None,
                        raw_tag=tag,
                        payload={
                            "tile136": int(elem.get("hai", "0")),
                            "tile": tile136_to_string(int(elem.get("hai", "0"))),
                        },
                    )
                )
                step_index += 1
            elif tag == "REACH":
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="reach",
                        actor=_parse_actor(tag, elem),
                        raw_tag=tag,
                        payload={
                            "who": int(elem.get("who", "0")),
                            "step": int(elem.get("step", "0")),
                            "ten": elem.get("ten"),
                        },
                    )
                )
                step_index += 1
            elif tag in {"AGARI", "RYUUKYOKU"}:
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type=tag.lower(),
                        actor=_parse_actor(tag, elem),
                        raw_tag=tag,
                        payload=dict(elem.attrib),
                    )
                )
                step_index += 1
            elif tag == "N":
                actor = int(elem.get("who", "0"))
                meld = decode_meld(actor=actor, value=int(elem.get("m", "0")))
                last_draws[actor] = None
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="meld",
                        actor=actor,
                        raw_tag=tag,
                        payload={
                            "kind": meld.kind,
                            "source": meld.source,
                            "called_tile": meld.called_tile,
                            "consumed_tiles": list(meld.consumed_tiles),
                            "raw_value": meld.raw_value,
                        },
                    )
                )
                step_index += 1
            elif tag and normalized_head in DRAW_TAG_TO_ACTOR and tag[1:].isdigit():
                tile136 = int(tag[1:])
                last_draws[DRAW_TAG_TO_ACTOR[normalized_head]] = tile136
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="draw",
                        actor=DRAW_TAG_TO_ACTOR[normalized_head],
                        raw_tag=tag,
                        payload={
                            "tile136": tile136,
                            "tile": tile136_to_string(tile136),
                        },
                    )
                )
                step_index += 1
            elif tag and normalized_head in DISCARD_TAG_TO_ACTOR and tag[1:].isdigit():
                tile136 = int(tag[1:])
                events.append(
                    Event(
                        game_id=game_id,
                        kyoku_index=kyoku_index,
                        step_index=step_index,
                        event_type="discard",
                        actor=DISCARD_TAG_TO_ACTOR[normalized_head],
                        raw_tag=tag,
                        payload={
                            "tile136": tile136,
                            "tile": tile136_to_string(tile136),
                            "tsumogiri": last_draws[DISCARD_TAG_TO_ACTOR[normalized_head]] == tile136,
                        },
                    )
                )
                step_index += 1

                last_draws[DISCARD_TAG_TO_ACTOR[normalized_head]] = None

            elem.clear()


        return events


def _parse_tiles_attr(value: str | None) -> list[str]:
    """解析输入数据并返回结构化结果。
    
    参数:
        value (str | None): 输入值。
    
    返回:
        list[str]: 列表结果。
    
    异常:
        无。
    """
    if not value:
        return []
    return [tile136_to_string(int(tile_id)) for tile_id in value.split(",") if tile_id]


def _parse_seed_dora_indicator(value: str | None) -> int:
    """解析输入数据并返回结构化结果。
    
    参数:
        value (str | None): 输入值。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    if not value:
        return 0
    parts = [part for part in value.split(",") if part]
    if len(parts) < 6:
        return 0
    return int(parts[5])


def _parse_round_wind(value: str | None) -> int:
    """Parse the round wind from Tenhou INIT seed."""
    if not value:
        return 0
    parts = [part for part in value.split(",") if part]
    if not parts:
        return 0
    return max(0, int(parts[0]) // 4)


def _parse_actor(tag: str, elem) -> int | None:
    """解析输入数据并返回结构化结果。
    
    参数:
        tag (str): 函数执行所需的输入参数。
        elem (Any): 函数执行所需的输入参数。
    
    返回:
        int | None: 函数执行结果。
    
    异常:
        无。
    """
    if tag in {"AGARI", "RYUUKYOKU"}:
        who = elem.get("who")
        if who is not None:
            return int(who)
    if tag == "REACH":
        who = elem.get("who")
        if who is not None:
            return int(who)
    return None
