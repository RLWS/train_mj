from __future__ import annotations

from dataclasses import asdict

from mjai.data.schema.events import Event
from mjai.data.schema.samples import TurnSample
from mjai.rules.legal_actions import legal_discards
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
                label_action = {"kind": "discard", "tile": event.payload["tile"]}
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
