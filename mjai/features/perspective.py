from __future__ import annotations


def relative_state(state: dict) -> dict:
    """Copy public state into self/next/opposite/previous seat order.

    Raw JSON and returned actions keep platform seats. Only encoder inputs rotate.
    Wind indices are compass directions, not player indices, and never rotate.
    """
    actor = int(state.get("actor", 0))
    result = dict(state)
    result["actor"] = 0
    for key in ("dealer", "last_discard_actor"):
        if state.get(key) is not None:
            result[key] = (int(state[key]) - actor) % 4
    for key in ("all_discards", "scores", "riichi_declared_by_player", "riichi_turn_by_player"):
        if key in state:
            rows = list(state[key])
            if len(rows) == 4:
                result[key] = [rows[(actor + i) % 4] for i in range(4)]
    if "all_melds" in state:
        result["all_melds"] = [
            [relative_action(meld, actor) for meld in state["all_melds"][(actor + i) % 4]]
            for i in range(4)
        ]
    result["own_melds"] = [relative_action(meld, actor) for meld in state.get("own_melds", [])]
    for key in ("discard_history", "public_events"):
        if key in state:
            result[key] = []
            for item in state[key]:
                entry = dict(item) if isinstance(item, dict) else {"actor": item[0], "tile": item[1]}
                entry = relative_action(entry, actor)
                if entry.get("actor") is not None:
                    entry["actor"] = (int(entry["actor"]) - actor) % 4
                result[key].append(entry)
    if "response_target" in state:
        result["response_target"] = relative_action(state["response_target"], actor)
    return result


def relative_action(action: dict, actor: int) -> dict:
    result = dict(action)
    if action.get("source") is not None:
        result["source"] = (int(action["source"]) - actor) % 4
    return result
