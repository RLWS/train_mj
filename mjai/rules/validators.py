from __future__ import annotations

from dataclasses import dataclass

from mjai.data.schema.events import Event
from mjai.rules.state import GameState
from mjai.rules.tiles import tile_to_base_index


@dataclass(slots=True, frozen=True)
class ReplaySummary:
    """Small replay summary for tests and diagnostics."""

    game_id: str
    kyokus_seen: int
    total_events: int
    total_discards: int
    total_melds: int


class ReplayValidator:
    """Validate parsed replay events against a minimal four-player state."""

    def validate_events(self, events: list[Event]) -> ReplaySummary:
        """回放并校验规范化事件序列。
        
        参数:
            self (Self): 当前对象实例。
            events (list[Event]): 规范化事件列表。
        
        返回:
            ReplaySummary: 函数执行结果。
        
        异常:
            ValueError: 当输入数据不满足函数约束时抛出。
        """
        if not events:
            raise ValueError("cannot validate an empty event list")

        state: GameState | None = None
        kyokus_seen = 0
        total_discards = 0
        total_melds = 0

        for event in events:
            if event.event_type == "init":
                state = GameState.from_init_payload(event.kyoku_index, event.payload)
                kyokus_seen += 1
                self._validate_init_state(state)
                continue

            if state is None:
                continue

            if event.event_type == "draw":
                assert event.actor is not None
                state.apply_draw(event.actor, event.payload["tile"])
                continue

            if event.event_type == "discard":
                assert event.actor is not None
                self._validate_discard_is_legal(state, event.actor, event.payload["tile"])
                state.apply_discard(event.actor, event.payload["tile"])
                total_discards += 1
                continue

            if event.event_type == "meld":
                assert event.actor is not None
                payload = event.payload
                self._validate_meld_shape(state, event.actor, payload)
                state.apply_meld(
                    actor=event.actor,
                    kind=payload["kind"],
                    source=payload["source"],
                    called_tile=payload["called_tile"],
                    consumed_tiles=payload["consumed_tiles"],
                )
                total_melds += 1
                continue

            if event.event_type == "reach":
                assert event.actor is not None
                state.apply_reach(event.actor, event.payload["step"])

        return ReplaySummary(
            game_id=events[0].game_id,
            kyokus_seen=kyokus_seen,
            total_events=len(events),
            total_discards=total_discards,
            total_melds=total_melds,
        )

    def _validate_init_state(self, state: GameState) -> None:
        """校验输入数据的一致性和合法性。
        
        参数:
            self (Self): 当前对象实例。
            state (GameState): 游戏状态或样本状态。
        
        返回:
            None: 无返回值。
        
        异常:
            ValueError: 当输入数据不满足函数约束时抛出。
        """
        for actor, hand in enumerate(state.hands):
            if hand.total_tiles() != 13:
                raise ValueError(
                    f"init hand must contain 13 tiles: kyoku={state.kyoku_index} actor={actor} "
                    f"count={hand.total_tiles()}"
                )

    def _validate_discard_is_legal(self, state: GameState, actor: int, tile: str) -> None:
        """校验输入数据的一致性和合法性。
        
        参数:
            self (Self): 当前对象实例。
            state (GameState): 游戏状态或样本状态。
            actor (int): 玩家座位索引。
            tile (str): 精确牌字符串。
        
        返回:
            None: 无返回值。
        
        异常:
            ValueError: 当输入数据不满足函数约束时抛出。
        """
        if not state.hands[actor].has_tile(tile):
            raise ValueError(
                f"illegal discard: kyoku={state.kyoku_index} actor={actor} tile={tile} "
                f"hand_snapshot={state.hands[actor].snapshot()}"
            )

    def _validate_meld_shape(self, state: GameState, actor: int, payload: dict) -> None:
        """校验输入数据的一致性和合法性。
        
        参数:
            self (Self): 当前对象实例。
            state (GameState): 游戏状态或样本状态。
            actor (int): 玩家座位索引。
            payload (dict): 函数执行所需的输入参数。
        
        返回:
            None: 无返回值。
        
        异常:
            ValueError: 当输入数据不满足函数约束时抛出。
        """
        kind = payload["kind"]
        called_tile = payload["called_tile"]
        consumed_tiles = payload["consumed_tiles"]

        if kind == "chi":
            if payload["source"] != (actor + 3) % 4:
                raise ValueError(f"chi source must be upper player: actor={actor} source={payload['source']}")
            if len(consumed_tiles) != 2:
                raise ValueError(f"chi must consume 2 tiles: {consumed_tiles}")

        if kind == "pon":
            if len(consumed_tiles) != 2:
                raise ValueError(f"pon must consume 2 tiles: {consumed_tiles}")

        if kind == "daiminkan":
            if len(consumed_tiles) != 3:
                raise ValueError(f"daiminkan must consume 3 tiles: {consumed_tiles}")

        if kind == "ankan":
            if len(consumed_tiles) != 4:
                raise ValueError(f"ankan must consume 4 tiles: {consumed_tiles}")
            if any(tile_to_base_index(tile) != tile_to_base_index(called_tile) for tile in consumed_tiles):
                raise ValueError(f"ankan tiles must share one base tile: {consumed_tiles}")

        if kind == "kakan":
            if len(consumed_tiles) != 3:
                raise ValueError(f"kakan must carry 3 tiles in current representation: {consumed_tiles}")

        if kind in {"chi", "pon", "daiminkan", "ankan"}:
            for tile in consumed_tiles:
                if not state.hands[actor].has_tile(tile):
                    raise ValueError(
                        f"meld consumes missing tile: kyoku={state.kyoku_index} actor={actor} "
                        f"kind={kind} tile={tile} hand_snapshot={state.hands[actor].snapshot()}"
                    )
