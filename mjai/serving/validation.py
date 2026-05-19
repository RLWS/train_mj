from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mjai.features.action_features import RESPONSE_ACTION_KINDS, TURN_ACTION_KINDS
from mjai.rules.tiles import tile_to_base_index


class ServingError(ValueError):
    code = "invalid_request"
    status_code = 400

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class RequestValidationError(ServingError):
    code = "invalid_state"
    status_code = 422


class BadRequestError(ServingError):
    code = "invalid_request"
    status_code = 400


class CandidateLimitError(ServingError):
    code = "too_many_candidates"
    status_code = 413


class ServiceUnavailableError(ServingError):
    code = "service_unavailable"
    status_code = 503


def validate_and_normalize_state(task: str, state: Mapping[str, Any] | None) -> dict[str, Any]:
    if state is None:
        raise RequestValidationError("state must be provided")

    normalized: dict[str, Any] = dict(state)
    normalized["hand"] = _normalize_hand(normalized.get("hand"))
    normalized["all_discards"] = _normalize_tile_rows(normalized.get("all_discards"), "all_discards")
    normalized["own_melds"] = _normalize_meld_list(normalized.get("own_melds"), "own_melds")
    normalized["scores"] = _normalize_scores(normalized.get("scores"))
    normalized["dora_indicators"] = _normalize_tile_list(normalized.get("dora_indicators"), "dora_indicators")
    normalized["actor"] = _normalize_seat(normalized.get("actor", 0), "actor")
    normalized["dealer"] = _normalize_seat(normalized.get("dealer", 0), "dealer")
    normalized["seat_wind"] = _normalize_seat(normalized.get("seat_wind", 0), "seat_wind")
    normalized["turn_number"] = _normalize_int(normalized.get("turn_number", 0), "turn_number")
    normalized["remaining_draws_estimate"] = _normalize_int(
        normalized.get("remaining_draws_estimate", 0),
        "remaining_draws_estimate",
    )
    normalized["honba"] = _normalize_int(normalized.get("honba", 0), "honba")
    normalized["riichi_sticks"] = _normalize_int(normalized.get("riichi_sticks", 0), "riichi_sticks")
    normalized["riichi_declared"] = bool(normalized.get("riichi_declared", False))
    normalized["temporary_furiten"] = bool(normalized.get("temporary_furiten", False))
    normalized["riichi_furiten"] = bool(normalized.get("riichi_furiten", False))
    if "riichi_declared_by_player" in normalized:
        normalized["riichi_declared_by_player"] = _normalize_bool_list(
            normalized.get("riichi_declared_by_player"),
            "riichi_declared_by_player",
        )
    else:
        normalized["riichi_declared_by_player"] = [False] * 4
        normalized["riichi_declared_by_player"][int(normalized["actor"])] = normalized["riichi_declared"]
    normalized["riichi_turn_by_player"] = _normalize_optional_int_list(
        normalized.get("riichi_turn_by_player"),
        "riichi_turn_by_player",
    )
    normalized["discard_history"] = _normalize_discard_history(
        normalized.get("discard_history"),
        "discard_history",
    )

    last_discard_actor = normalized.get("last_discard_actor")
    normalized["last_discard_actor"] = (
        None if last_discard_actor is None else _normalize_seat(last_discard_actor, "last_discard_actor")
    )
    if "last_discard_tile" not in normalized:
        normalized["last_discard_tile"] = None
    elif normalized["last_discard_tile"] is not None:
        normalized["last_discard_tile"] = _normalize_tile(
            normalized["last_discard_tile"],
            "last_discard_tile",
        )

    if task == "response":
        normalized["all_melds"] = _normalize_player_melds(normalized.get("all_melds"), "all_melds")
        response_target = normalized.get("response_target")
        if not isinstance(response_target, Mapping):
            raise RequestValidationError(
                "response requests require state.response_target with source and tile"
            )
        response_target_normalized = dict(response_target)
        response_target_normalized["source"] = _normalize_seat(
            response_target_normalized.get("source"),
            "response_target.source",
        )
        if response_target_normalized.get("tile") is None:
            raise RequestValidationError("response_target.tile must be provided")
        response_target_normalized["tile"] = _normalize_tile(
            response_target_normalized["tile"],
            "response_target.tile",
        )
        normalized["response_target"] = response_target_normalized

    return normalized


def validate_legal_actions(
    legal_actions: list[Mapping[str, Any]] | None,
    *,
    candidate_limit: int,
    task: str | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(legal_actions, list):
        raise BadRequestError("legal_actions must be a list")
    if not legal_actions:
        raise BadRequestError("legal_actions must not be empty")
    if len(legal_actions) > candidate_limit:
        raise CandidateLimitError(
            f"legal_actions exceeds the configured limit: {len(legal_actions)} > {candidate_limit}"
        )
    normalized: list[dict[str, Any]] = []
    for index, action in enumerate(legal_actions):
        if not isinstance(action, Mapping):
            raise BadRequestError(f"legal_actions[{index}] must be an object")
        normalized.append(_normalize_action(dict(action), index=index, task=task))
    return normalized


def _normalize_hand(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RequestValidationError("hand must be an object")
    normalized: dict[str, int] = {}
    for tile, count in value.items():
        normalized[_normalize_tile(tile, "hand tile")] = _normalize_int(count, "hand count")
    return normalized


def _normalize_tile_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError(f"{field_name} must be a list")
    return [_normalize_tile(tile, field_name) for tile in value]


def _normalize_tile_rows(value: Any, field_name: str) -> list[list[str]]:
    if value is None:
        return [[], [], [], []]
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError(f"{field_name} must be a list of four lists")
    rows = list(value[:4])
    while len(rows) < 4:
        rows.append([])
    normalized_rows: list[list[str]] = []
    for index, row in enumerate(rows[:4]):
        if row is None:
            normalized_rows.append([])
            continue
        if not isinstance(row, (list, tuple)):
            raise RequestValidationError(f"{field_name}[{index}] must be a list")
        normalized_rows.append([_normalize_tile(tile, f"{field_name}[{index}]") for tile in row])
    return normalized_rows


def _normalize_meld_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError(f"{field_name} must be a list")
    normalized: list[dict[str, Any]] = []
    for index, meld in enumerate(value):
        if not isinstance(meld, Mapping):
            raise RequestValidationError(f"{field_name}[{index}] must be an object")
        normalized_meld = dict(meld)
        if normalized_meld.get("called_tile") is not None:
            normalized_meld["called_tile"] = _normalize_tile(
                normalized_meld["called_tile"],
                f"{field_name}[{index}].called_tile",
            )
        normalized_meld["consumed_tiles"] = _normalize_tile_list(
            normalized_meld.get("consumed_tiles"),
            f"{field_name}[{index}].consumed_tiles",
        )
        normalized.append(normalized_meld)
    return normalized


def _normalize_player_melds(value: Any, field_name: str) -> list[list[dict[str, Any]]]:
    if value is None:
        return [[], [], [], []]
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError(f"{field_name} must be a list of four lists")
    rows = list(value[:4])
    while len(rows) < 4:
        rows.append([])
    normalized_rows: list[list[dict[str, Any]]] = []
    for index, row in enumerate(rows[:4]):
        normalized_rows.append(_normalize_meld_list(row, f"{field_name}[{index}]"))
    return normalized_rows


def _normalize_bool_list(value: Any, field_name: str) -> list[bool]:
    if value is None:
        return [False] * 4
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError(f"{field_name} must be a list")
    rows = list(value[:4])
    while len(rows) < 4:
        rows.append(False)
    return [bool(item) for item in rows]


def _normalize_optional_int_list(value: Any, field_name: str) -> list[int | None]:
    if value is None:
        return [None] * 4
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError(f"{field_name} must be a list")
    rows = list(value[:4])
    while len(rows) < 4:
        rows.append(None)
    normalized: list[int | None] = []
    for item in rows:
        if item is None:
            normalized.append(None)
        else:
            normalized.append(_normalize_int(item, field_name))
    return normalized


def _normalize_discard_history(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError(f"{field_name} must be a list")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RequestValidationError(f"{field_name}[{index}] must be an object")
        actor = _normalize_seat(item.get("actor"), f"{field_name}[{index}].actor")
        tile = _normalize_tile(item.get("tile"), f"{field_name}[{index}].tile")
        normalized.append({"actor": actor, "tile": tile})
    return normalized


def _normalize_action(action: dict[str, Any], *, index: int, task: str | None) -> dict[str, Any]:
    kind = action.get("kind")
    if not isinstance(kind, str) or not kind:
        raise RequestValidationError(f"legal_actions[{index}].kind must be provided")
    allowed_kinds = _allowed_action_kinds(task)
    if kind not in allowed_kinds:
        raise RequestValidationError(f"legal_actions[{index}].kind is not supported: {kind}")
    if action.get("tile") is not None:
        action["tile"] = _normalize_tile(action["tile"], f"legal_actions[{index}].tile")
    if action.get("source") is not None:
        action["source"] = _normalize_seat(action["source"], f"legal_actions[{index}].source")
    if "consumed_tiles" in action:
        action["consumed_tiles"] = _normalize_tile_list(
            action.get("consumed_tiles"),
            f"legal_actions[{index}].consumed_tiles",
        )
    return action


def _allowed_action_kinds(task: str | None) -> set[str]:
    if task == "turn":
        return set(TURN_ACTION_KINDS) - {"unknown"}
    if task == "response":
        return set(RESPONSE_ACTION_KINDS) - {"unknown"}
    return (set(TURN_ACTION_KINDS) | set(RESPONSE_ACTION_KINDS)) - {"unknown"}


def _normalize_tile(value: Any, field_name: str) -> str:
    tile = str(value)
    try:
        tile_to_base_index(tile)
    except ValueError as exc:
        raise RequestValidationError(f"{field_name} must be a valid tile string") from exc
    return tile


def _normalize_scores(value: Any) -> list[int]:
    if value is None:
        return [0, 0, 0, 0]
    if not isinstance(value, (list, tuple)):
        raise RequestValidationError("scores must be a list of four numbers")
    scores = [int(item) for item in list(value)[:4]]
    while len(scores) < 4:
        scores.append(0)
    return scores


def _normalize_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{field_name} must be an integer") from exc


def _normalize_seat(value: Any, field_name: str) -> int:
    seat = _normalize_int(value, field_name)
    if seat < 0 or seat > 3:
        raise RequestValidationError(f"{field_name} must be in [0, 3]")
    return seat
