from __future__ import annotations

from dataclasses import asdict

from mjai.data.schema.events import Event
from mjai.data.schema.samples import TurnSample
from mjai.rules.legal_actions import Action, canonicalize_consumed, legal_discards, legal_responses
from mjai.rules.state import GameState


class TurnSampleBuilder:
    """Build turn discard samples from parsed replay events."""

    def build_samples(self, events: list[Event]) -> list[TurnSample]:
        state: GameState | None = None
        samples: list[TurnSample] = []

        for event in events:
            if event.event_type == "init":
                state = GameState.from_init_payload(event.hand_index, event.payload)
                continue

            if state is None:
                continue

            if event.event_type == "draw":
                assert event.actor is not None
                state.apply_draw(event.actor, event.payload["tile"])
                continue

            if event.event_type == "discard":
                assert event.actor is not None
                legal_actions = [asdict(action) for action in legal_discards(state.hands[event.actor])]
                label_action = asdict(Action(kind="discard", tile=event.payload["tile"]))
                sample = TurnSample(
                    sample_id=(
                        f"{event.game_id}:{event.hand_index}:{event.step_index}:"
                        f"{event.actor}:turn"
                    ),
                    game_id=event.game_id,
                    hand_index=event.hand_index,
                    step_index=event.step_index,
                    actor=event.actor,
                    decision_type="turn",
                    state=_player_view_snapshot(state, event.actor),
                    legal_actions=legal_actions,
                    label_action=label_action,
                )
                samples.append(sample)
                state.apply_discard(event.actor, event.payload["tile"])
                continue

            if event.event_type == "meld":
                assert event.actor is not None
                payload = event.payload
                state.apply_meld(
                    actor=event.actor,
                    kind=payload["kind"],
                    source=payload["source"],
                    called_tile=payload["called_tile"],
                    consumed_tiles=payload["consumed_tiles"],
                )
                continue

            if event.event_type == "reach":
                assert event.actor is not None
                state.apply_reach(event.actor, event.payload["step"])

        return samples


class ResponseSampleBuilder:
    """Build response samples after each discard.

    This first pass supports pass / chi / pon / daiminkan labels.
    Ron labels are skipped until a win checker is implemented.
    """

    def build_samples(self, events: list[Event]) -> list[TurnSample]:
        state: GameState | None = None
        samples: list[TurnSample] = []

        for event_index, event in enumerate(events):
            if event.event_type == "init":
                state = GameState.from_init_payload(event.hand_index, event.payload)
                continue

            if state is None:
                continue

            if event.event_type == "draw":
                assert event.actor is not None
                state.apply_draw(event.actor, event.payload["tile"])
                continue

            if event.event_type == "discard":
                assert event.actor is not None
                state.apply_discard(event.actor, event.payload["tile"])

                resolution = _resolve_discard_window(events, event_index)
                if resolution["kind"] == "ron":
                    continue

                for responder in range(4):
                    if responder == event.actor:
                        continue

                    # 其它三家对于此次弃牌，统计合法动作
                    legal_actions = [
                        asdict(action)
                        for action in legal_responses(
                            hand=state.hands[responder],
                            actor=responder,
                            source=event.actor,
                            tile=event.payload["tile"],
                        )
                    ]
                    label_action = _response_label_for_actor(responder, event, resolution)
                    sample = TurnSample(
                        sample_id=(
                            f"{event.game_id}:{event.hand_index}:{event.step_index}:"
                            f"{responder}:response"
                        ),
                        game_id=event.game_id,
                        hand_index=event.hand_index,
                        step_index=event.step_index,
                        actor=responder,
                        decision_type="response",
                        state=_response_view_snapshot(state, responder, event.actor, event.payload["tile"]),
                        legal_actions=legal_actions,
                        label_action=label_action,
                    )
                    samples.append(sample)
                continue

            if event.event_type == "meld":
                assert event.actor is not None
                payload = event.payload
                state.apply_meld(
                    actor=event.actor,
                    kind=payload["kind"],
                    source=payload["source"],
                    called_tile=payload["called_tile"],
                    consumed_tiles=payload["consumed_tiles"],
                )
                continue

            if event.event_type == "reach":
                assert event.actor is not None
                state.apply_reach(event.actor, event.payload["step"])

        return samples


def _player_view_snapshot(state: GameState, actor: int) -> dict:
    return {
        "actor": actor,
        "hand": state.hands[actor].snapshot(),
        "own_discards": list(state.discards[actor]),
        "own_melds": [
            {
                "kind": meld.kind,
                "source": meld.source,
                "called_tile": meld.called_tile,
                "consumed_tiles": list(meld.consumed_tiles),
            }
            for meld in state.melds[actor]
        ],
        "riichi_declared": state.riichi_declared[actor],
        "last_discard_actor": state.last_discard_actor,
        "last_discard_tile": state.last_discard_tile,
        "all_discards": [list(discard_row) for discard_row in state.discards],
    }


def _response_view_snapshot(state: GameState, actor: int, source: int, tile: str) -> dict:
    snapshot = _player_view_snapshot(state, actor)
    snapshot["response_target"] = {"source": source, "tile": tile}
    return snapshot


def _resolve_discard_window(events: list[Event], discard_index: int) -> dict:
    discard_event = events[discard_index]

    for future in events[discard_index + 1 :]:
        if future.hand_index != discard_event.hand_index:
            break

        if future.event_type == "reach":
            continue

        if future.event_type == "dora":
            continue

        if future.event_type == "meld":
            return {
                "kind": "meld",
                "actor": future.actor,
                "action": {
                    "kind": future.payload["kind"],
                    "tile": future.payload["called_tile"],
                    "consumed_tiles": canonicalize_consumed(tuple(future.payload["consumed_tiles"])),
                    "source": future.payload["source"],
                },
            }

        if future.event_type == "agari":
            if future.payload.get("fromWho") == str(discard_event.actor):
                return {"kind": "ron"}
            return {"kind": "pass"}

        if future.event_type in {"draw", "discard", "init", "ryuukyoku"}:
            return {"kind": "pass"}

    return {"kind": "pass"}


def _response_label_for_actor(responder: int, discard_event: Event, resolution: dict) -> dict:
    if resolution["kind"] == "meld" and resolution["actor"] == responder:
        return resolution["action"]
    return asdict(Action(kind="pass"))
