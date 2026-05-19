from __future__ import annotations

from dataclasses import asdict

from mjai.data.schema.events import Event
from mjai.data.schema.samples import TurnSample
from mjai.rules.legal_actions import (
    Action,
    canonicalize_consumed,
    legal_responses,
    legal_turn_actions,
)
from mjai.rules.state import GameState


class TurnSampleBuilder:
    """Build turn decision samples from parsed replay events."""

    def build_samples(self, events: list[Event]) -> list[TurnSample]:
        """构建与函数名对应的数据结构或处理结果。
        
        参数:
            self (Self): 当前对象实例。
            events (list[Event]): 规范化事件列表。
        
        返回:
            list[TurnSample]: 列表结果。
        
        异常:
            无。
        """
        state: GameState | None = None
        samples: list[TurnSample] = []
        pending_turn_labels: dict[int, str] = {}
        last_draw_actor: int | None = None

        for event in events:
            if event.event_type == "init":
                state = GameState.from_init_payload(event.kyoku_index, event.payload)
                pending_turn_labels = {}
                last_draw_actor = None
                continue

            if state is None:
                continue

            if event.event_type == "draw":
                assert event.actor is not None
                state.apply_draw(event.actor, event.payload["tile"])
                last_draw_actor = event.actor
                continue

            if event.event_type == "discard":
                assert event.actor is not None
                label_kind = pending_turn_labels.pop(event.actor, "discard")
                legal_actions = [
                    asdict(action)
                    for action in legal_turn_actions(
                        state.hands[event.actor],
                        state=state,
                        actor=event.actor,
                    )
                ]
                label_action = asdict(Action(kind=label_kind, tile=event.payload["tile"]))
                sample = TurnSample(
                    sample_id=(
                        f"{event.game_id}:{event.kyoku_index}:{event.step_index}:"
                        f"{event.actor}:turn"
                    ),
                    game_id=event.game_id,
                    kyoku_index=event.kyoku_index,
                    step_index=event.step_index,
                    actor=event.actor,
                    decision_type="turn",
                    state=_player_view_snapshot(state, event.actor),
                    legal_actions=legal_actions,
                    label_action=label_action,
                )
                samples.append(sample)
                state.apply_discard(event.actor, event.payload["tile"])
                last_draw_actor = None
                continue

            if event.event_type == "meld":
                assert event.actor is not None
                payload = event.payload
                if payload["kind"] in {"ankan", "kakan"}:
                    sample = _build_turn_sample(
                        state=state,
                        event=event,
                        actor=event.actor,
                        label_action=asdict(
                            Action(
                                kind=payload["kind"],
                                tile=payload["called_tile"],
                                consumed_tiles=canonicalize_consumed(tuple(payload["consumed_tiles"])),
                                source=payload["source"],
                            )
                        ),
                    )
                    samples.append(sample)
                state.apply_meld(
                    actor=event.actor,
                    kind=payload["kind"],
                    source=payload["source"],
                    called_tile=payload["called_tile"],
                    consumed_tiles=payload["consumed_tiles"],
                )
                last_draw_actor = None
                continue

            if event.event_type == "dora":
                state.add_dora_indicator(event.payload["tile"])
                continue

            if event.event_type == "reach":
                assert event.actor is not None
                if event.payload["step"] == 1:
                    pending_turn_labels[event.actor] = "riichi_discard"
                    continue
                state.apply_reach(event.actor, event.payload["step"])
                continue

            if event.event_type == "agari":
                assert event.actor is not None
                if event.payload.get("who") == event.payload.get("fromWho"):
                    sample = _build_turn_sample(
                        state=state,
                        event=event,
                        actor=event.actor,
                        label_action=asdict(Action(kind="tsumo")),
                    )
                    samples.append(sample)
                continue

            if event.event_type == "ryuukyoku" and event.payload.get("type") == "yao9":
                actor = event.actor if event.actor is not None else last_draw_actor
                if actor is None:
                    continue
                sample = _build_turn_sample(
                    state=state,
                    event=event,
                    actor=actor,
                    label_action=asdict(Action(kind="kyuushu_kyuuhai")),
                )
                samples.append(sample)

        return samples


class ResponseSampleBuilder:
    """Build response samples after each discard.

    This pass supports pass / chi / pon / daiminkan / ron / chankan labels.
    """

    def build_samples(self, events: list[Event]) -> list[TurnSample]:
        """构建与函数名对应的数据结构或处理结果。
        
        参数:
            self (Self): 当前对象实例。
            events (list[Event]): 规范化事件列表。
        
        返回:
            list[TurnSample]: 列表结果。
        
        异常:
            无。
        """
        state: GameState | None = None
        samples: list[TurnSample] = []

        for event_index, event in enumerate(events):
            if event.event_type == "init":
                state = GameState.from_init_payload(event.kyoku_index, event.payload)
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
                for responder in range(4):
                    if responder == event.actor:
                        continue

                    legal_action_objects = legal_responses(
                        hand=state.hands[responder],
                        actor=responder,
                        source=event.actor,
                        tile=event.payload["tile"],
                        state=state,
                    )
                    legal_actions = [asdict(action) for action in legal_action_objects]
                    label_action = _response_label_for_actor(responder, event, resolution)
                    sample = TurnSample(
                        sample_id=(
                            f"{event.game_id}:{event.kyoku_index}:{event.step_index}:"
                            f"{responder}:response"
                        ),
                        game_id=event.game_id,
                        kyoku_index=event.kyoku_index,
                        step_index=event.step_index,
                        actor=responder,
                        decision_type="response",
                        state=_response_view_snapshot(state, responder, event.actor, event.payload["tile"]),
                        legal_actions=legal_actions,
                        label_action=label_action,
                    )
                    samples.append(sample)
                    _mark_missed_ron_if_needed(
                        state=state,
                        responder=responder,
                        source=event.actor,
                        tile=event.payload["tile"],
                        label_action=label_action,
                        legal_actions=legal_action_objects,
                    )
                continue

            if event.event_type == "meld":
                assert event.actor is not None
                payload = event.payload
                if payload["kind"] == "kakan":
                    resolution = _resolve_kakan_window(events, event_index)
                    for responder in range(4):
                        if responder == event.actor:
                            continue

                        legal_action_objects = legal_responses(
                            hand=state.hands[responder],
                            actor=responder,
                            source=event.actor,
                            tile=payload["called_tile"],
                            state=state,
                            allow_chankan=True,
                        )
                        legal_actions = [asdict(action) for action in legal_action_objects]
                        label_action = _kakan_response_label_for_actor(
                            responder=responder,
                            actor=event.actor,
                            tile=payload["called_tile"],
                            resolution=resolution,
                        )
                        sample = TurnSample(
                            sample_id=(
                                f"{event.game_id}:{event.kyoku_index}:{event.step_index}:"
                                f"{responder}:response"
                            ),
                            game_id=event.game_id,
                            kyoku_index=event.kyoku_index,
                            step_index=event.step_index,
                            actor=responder,
                            decision_type="response",
                            state=_response_view_snapshot(state, responder, event.actor, payload["called_tile"]),
                            legal_actions=legal_actions,
                            label_action=label_action,
                        )
                        samples.append(sample)
                        _mark_missed_chankan_if_needed(
                            state=state,
                            responder=responder,
                            source=event.actor,
                            tile=payload["called_tile"],
                            label_action=label_action,
                            legal_actions=legal_action_objects,
                        )
                    if resolution["kind"] == "chankan":
                        continue
                state.apply_meld(
                    actor=event.actor,
                    kind=payload["kind"],
                    source=payload["source"],
                    called_tile=payload["called_tile"],
                    consumed_tiles=payload["consumed_tiles"],
                )
                continue

            if event.event_type == "dora":
                state.add_dora_indicator(event.payload["tile"])
                continue

            if event.event_type == "reach":
                assert event.actor is not None
                state.apply_reach(event.actor, event.payload["step"])

        return samples


def _player_view_snapshot(state: GameState, actor: int) -> dict:
    """执行与函数名对应的业务逻辑。
    
    参数:
        state (GameState): 游戏状态或样本状态。
        actor (int): 玩家座位索引。
    
    返回:
        dict: 字典结果。
    
    异常:
        无。
    """
    own_draws = len(state.discards[actor]) + (1 if state.hands[actor].total_tiles() % 3 == 2 else 0)
    return {
        "actor": actor,
        "hand": state.hands[actor].snapshot(),
        "scores": list(state.scores),
        "dealer": state.dealer,
        "seat_wind": (actor - state.dealer) % 4,
        "honba": state.honba,
        "riichi_sticks": state.riichi_sticks,
        "dora_indicators": list(state.dora_indicators),
        "dora_base_indices": sorted(state.dora_base_indices()),
        "turn_number": own_draws,
        "remaining_draws_estimate": max(0, 18 - own_draws),
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
        "all_melds": [
            [
                {
                    "kind": meld.kind,
                    "source": meld.source,
                    "called_tile": meld.called_tile,
                    "consumed_tiles": list(meld.consumed_tiles),
                }
                for meld in player_melds
            ]
            for player_melds in state.melds
        ],
        "riichi_declared": state.riichi_declared[actor],
        "riichi_declared_by_player": list(state.riichi_declared),
        "riichi_turn_by_player": list(state.riichi_turns),
        "temporary_furiten": state.temporary_furiten[actor],
        "riichi_furiten": state.riichi_furiten[actor],
        "last_discard_actor": state.last_discard_actor,
        "last_discard_tile": state.last_discard_tile,
        "all_discards": [list(discard_row) for discard_row in state.discards],
        "discard_history": [
            {"actor": discard_actor, "tile": tile}
            for discard_actor, tile in state.discard_history
        ],
    }


def _response_view_snapshot(state: GameState, actor: int, source: int, tile: str) -> dict:
    """执行与函数名对应的业务逻辑。
    
    参数:
        state (GameState): 游戏状态或样本状态。
        actor (int): 玩家座位索引。
        source (int): 动作来源玩家座位索引。
        tile (str): 精确牌字符串。
    
    返回:
        dict: 字典结果。
    
    异常:
        无。
    """
    snapshot = _player_view_snapshot(state, actor)
    snapshot["response_target"] = {"source": source, "tile": tile}
    return snapshot


def _build_turn_sample(
    state: GameState,
    event: Event,
    actor: int,
    label_action: dict,
) -> TurnSample:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        state (GameState): 游戏状态或样本状态。
        event (Event): 单个规范化事件。
        actor (int): 玩家座位索引。
        label_action (dict): 标签动作字典。
    
    返回:
        TurnSample: 函数执行结果。
    
    异常:
        无。
    """
    legal_actions = [
        asdict(action)
        for action in legal_turn_actions(
            state.hands[actor],
            state=state,
            actor=actor,
        )
    ]
    return TurnSample(
        sample_id=f"{event.game_id}:{event.kyoku_index}:{event.step_index}:{actor}:turn",
        game_id=event.game_id,
        kyoku_index=event.kyoku_index,
        step_index=event.step_index,
        actor=actor,
        decision_type="turn",
        state=_player_view_snapshot(state, actor),
        legal_actions=legal_actions,
        label_action=label_action,
    )


def _resolve_discard_window(events: list[Event], discard_index: int) -> dict:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        events (list[Event]): 规范化事件列表。
        discard_index (int): 相关索引值。
    
    返回:
        dict: 字典结果。
    
    异常:
        无。
    """
    discard_event = events[discard_index]

    for future in events[discard_index + 1 :]:
        if future.kyoku_index != discard_event.kyoku_index:
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
                return {"kind": "ron", "actor": int(future.payload["who"])}
            return {"kind": "pass"}

        if future.event_type in {"draw", "discard", "init", "ryuukyoku"}:
            return {"kind": "pass"}

    return {"kind": "pass"}


def _resolve_kakan_window(events: list[Event], meld_index: int) -> dict:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        events (list[Event]): 规范化事件列表。
        meld_index (int): 相关索引值。
    
    返回:
        dict: 字典结果。
    
    异常:
        无。
    """
    meld_event = events[meld_index]

    for future in events[meld_index + 1 :]:
        if future.kyoku_index != meld_event.kyoku_index:
            break

        if future.event_type == "reach":
            continue

        if future.event_type == "dora":
            continue

        if future.event_type == "agari" and future.payload.get("fromWho") == str(meld_event.actor):
            return {"kind": "chankan", "actor": int(future.payload["who"])}

        if future.event_type in {"draw", "discard", "init", "ryuukyoku", "meld"}:
            return {"kind": "pass"}

    return {"kind": "pass"}


def _response_label_for_actor(responder: int, discard_event: Event, resolution: dict) -> dict:
    """执行与函数名对应的业务逻辑。
    
    参数:
        responder (int): 响应玩家座位索引。
        discard_event (Event): 函数执行所需的输入参数。
        resolution (dict): 函数执行所需的输入参数。
    
    返回:
        dict: 字典结果。
    
    异常:
        无。
    """
    if resolution["kind"] == "ron" and resolution["actor"] == responder:
        return asdict(Action(kind="ron", tile=discard_event.payload["tile"], source=discard_event.actor))
    if resolution["kind"] == "meld" and resolution["actor"] == responder:
        return resolution["action"]
    return asdict(Action(kind="pass"))


def _kakan_response_label_for_actor(responder: int, actor: int, tile: str, resolution: dict) -> dict:
    """执行与函数名对应的业务逻辑。
    
    参数:
        responder (int): 响应玩家座位索引。
        actor (int): 玩家座位索引。
        tile (str): 精确牌字符串。
        resolution (dict): 函数执行所需的输入参数。
    
    返回:
        dict: 字典结果。
    
    异常:
        无。
    """
    if resolution["kind"] == "chankan" and resolution["actor"] == responder:
        return asdict(Action(kind="chankan", tile=tile, source=actor))
    return asdict(Action(kind="pass"))


def _mark_missed_ron_if_needed(
    state: GameState,
    responder: int,
    source: int,
    tile: str,
    label_action: dict,
    legal_actions: list[Action],
) -> None:
    """根据业务规则标记相关状态。
    
    参数:
        state (GameState): 游戏状态或样本状态。
        responder (int): 响应玩家座位索引。
        source (int): 动作来源玩家座位索引。
        tile (str): 精确牌字符串。
        label_action (dict): 标签动作字典。
        legal_actions (list[Action]): 合法动作列表。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    if label_action["kind"] != "pass":
        return

    if responder == source:
        return

    if any(action.kind == "ron" for action in legal_actions):
        state.mark_missed_ron(responder)


def _mark_missed_chankan_if_needed(
    state: GameState,
    responder: int,
    source: int,
    tile: str,
    label_action: dict,
    legal_actions: list[Action],
) -> None:
    """根据业务规则标记相关状态。
    
    参数:
        state (GameState): 游戏状态或样本状态。
        responder (int): 响应玩家座位索引。
        source (int): 动作来源玩家座位索引。
        tile (str): 精确牌字符串。
        label_action (dict): 标签动作字典。
        legal_actions (list[Action]): 合法动作列表。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    if label_action["kind"] != "pass":
        return

    if any(action.kind == "chankan" for action in legal_actions):
        state.mark_missed_ron(responder)
