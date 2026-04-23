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
        file_path = Path(path)
        game_id = file_path.stem
        hand_index = -1
        step_index = 0
        events: list[Event] = []

        for _, elem in iterparse(file_path, events=("end",)):
            tag = elem.tag
            tag_head = tag[0] if tag else ""
            normalized_head = tag_head.upper()

            if tag == "GO":
                events.append(
                    Event(
                        game_id=game_id,
                        hand_index=hand_index,
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
                        hand_index=hand_index,
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
                        hand_index=hand_index,
                        step_index=step_index,
                        event_type="taikyoku",
                        actor=None,
                        raw_tag=tag,
                        payload={"oya": int(elem.get("oya", "0"))},
                    )
                )
                step_index += 1
            elif tag == "INIT":
                hand_index += 1
                step_index = 0
                events.append(
                    Event(
                        game_id=game_id,
                        hand_index=hand_index,
                        step_index=step_index,
                        event_type="init",
                        actor=None,
                        raw_tag=tag,
                        payload={
                            "seed": elem.get("seed"),
                            "ten": elem.get("ten"),
                            "oya": int(elem.get("oya", "0")),
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
                        hand_index=hand_index,
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
                        hand_index=hand_index,
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
                        hand_index=hand_index,
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
                events.append(
                    Event(
                        game_id=game_id,
                        hand_index=hand_index,
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
                events.append(
                    Event(
                        game_id=game_id,
                        hand_index=hand_index,
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
                        hand_index=hand_index,
                        step_index=step_index,
                        event_type="discard",
                        actor=DISCARD_TAG_TO_ACTOR[normalized_head],
                        raw_tag=tag,
                        payload={
                            "tile136": tile136,
                            "tile": tile136_to_string(tile136),
                            "tsumogiri": tag[0].islower(),
                        },
                    )
                )
                step_index += 1

            elem.clear()


        return events


def _parse_tiles_attr(value: str | None) -> list[str]:
    if not value:
        return []
    return [tile136_to_string(int(tile_id)) for tile_id in value.split(",") if tile_id]


def _parse_actor(tag: str, elem) -> int | None:
    if tag in {"AGARI", "RYUUKYOKU"}:
        who = elem.get("who")
        if who is not None:
            return int(who)
    if tag == "REACH":
        who = elem.get("who")
        if who is not None:
            return int(who)
    return None
