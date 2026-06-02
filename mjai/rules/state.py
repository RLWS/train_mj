from __future__ import annotations

from dataclasses import dataclass, field

from mjai.rules.tiles import AKA_BASE_INDICES, base_index_to_tile, dora_from_indicator, parse_tile, tile_to_base_index

AKA_TILES_BY_INDEX = {
    4: "0m",
    13: "0p",
    22: "0s",
}


@dataclass(slots=True)
class HandState:
    """Minimal hand state for local rule validation and legal discard generation."""

    counts: list[int] = field(default_factory=lambda: [0] * 34)
    aka_counts: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_tiles(cls, tiles: list[str]) -> "HandState":
        """执行与函数名对应的业务逻辑。
        
        参数:
            cls (type[Self]): 当前类对象。
            tiles (list[str]): 精确牌字符串列表。
        
        返回:
            'HandState': 函数执行结果。
        
        异常:
            无。
        """
        hand = cls()
        for tile in tiles:
            hand.add_tile(tile)
        return hand

    def copy(self) -> "HandState":
        """复制当前对象或数据。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            'HandState': 函数执行结果。
        
        异常:
            无。
        """
        return HandState(counts=self.counts.copy(), aka_counts=self.aka_counts.copy())

    def total_tiles(self) -> int:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            int: 整数结果。
        
        异常:
            无。
        """
        return sum(self.counts)

    def count_exact(self, tile: str) -> int:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            tile (str): 精确牌字符串。
        
        返回:
            int: 整数结果。
        
        异常:
            无。
        """
        parsed = parse_tile(tile)
        if parsed.is_aka:
            return self.aka_counts.get(parsed.base_index, 0)
        if parsed.base_index in AKA_BASE_INDICES:
            return self.counts[parsed.base_index] - self.aka_counts.get(parsed.base_index, 0)
        return self.counts[parsed.base_index]

    def add_tile(self, tile: str) -> None:
        """向当前状态添加输入数据。
        
        参数:
            self (Self): 当前对象实例。
            tile (str): 精确牌字符串。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        parsed = parse_tile(tile)
        self.counts[parsed.base_index] += 1
        if parsed.is_aka:
            self.aka_counts[parsed.base_index] = self.aka_counts.get(parsed.base_index, 0) + 1

    def remove_tile(self, tile: str) -> None:
        """从当前状态移除输入数据。
        
        参数:
            self (Self): 当前对象实例。
            tile (str): 精确牌字符串。
        
        返回:
            None: 无返回值。
        
        异常:
            ValueError: 当输入数据不满足函数约束时抛出。
        """
        parsed = parse_tile(tile)
        if self.count_exact(tile) <= 0:
            raise ValueError(f"tile not available in hand: {tile}")

        self.counts[parsed.base_index] -= 1
        if parsed.is_aka:
            remaining = self.aka_counts.get(parsed.base_index, 0) - 1
            if remaining < 0:
                raise ValueError(f"red tile bookkeeping underflow: {tile}")
            if remaining == 0:
                self.aka_counts.pop(parsed.base_index, None)
            else:
                self.aka_counts[parsed.base_index] = remaining

    def has_tile(self, tile: str) -> bool:
        """判断与函数名对应的状态是否存在。
        
        参数:
            self (Self): 当前对象实例。
            tile (str): 精确牌字符串。
        
        返回:
            bool: 判断结果。
        
        异常:
            无。
        """
        return self.count_exact(tile) > 0

    def exact_tiles_for_base(self, base_index: int) -> list[str]:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            base_index (int): 三十四种牌基础索引。
        
        返回:
            list[str]: 列表结果。
        
        异常:
            无。
        """
        count = self.counts[base_index]
        if count <= 0:
            return []

        red_count = self.aka_counts.get(base_index, 0)
        exact_tiles: list[str] = []
        if red_count:
            exact_tiles.extend([AKA_TILES_BY_INDEX[base_index]] * red_count)
        normal_count = count - red_count
        if normal_count > 0:
            exact_tiles.extend([base_index_to_tile(base_index)] * normal_count)
        return exact_tiles

    def snapshot(self) -> dict[str, int]:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            dict[str, int]: 字典结果。
        
        异常:
            无。
        """
        tiles: dict[str, int] = {}
        for index, count in enumerate(self.counts):
            if count <= 0:
                continue
            red_count = self.aka_counts.get(index, 0)
            if red_count:
                tiles[AKA_TILES_BY_INDEX[index]] = red_count
            normal_count = count - red_count
            if normal_count > 0:
                tiles[base_index_to_tile(index)] = normal_count
        return tiles


@dataclass(slots=True, frozen=True)
class MeldState:
    """Minimal meld state needed for replay validation."""

    kind: str
    source: int
    called_tile: str
    consumed_tiles: tuple[str, ...]

    @property
    def base_index(self) -> int:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            int: 整数结果。
        
        异常:
            无。
        """
        return tile_to_base_index(self.called_tile)


@dataclass(slots=True)
class GameState:
    """Four-player replay state for parsed Tenhou events."""

    hands: list[HandState] = field(default_factory=lambda: [HandState() for _ in range(4)])
    discards: list[list[str]] = field(default_factory=lambda: [[] for _ in range(4)])
    melds: list[list[MeldState]] = field(default_factory=lambda: [[] for _ in range(4)])
    riichi_declared: list[bool] = field(default_factory=lambda: [False] * 4)
    riichi_turns: list[int | None] = field(default_factory=lambda: [None] * 4)
    riichi_pending: list[bool] = field(default_factory=lambda: [False] * 4)
    temporary_furiten: list[bool] = field(default_factory=lambda: [False] * 4)
    riichi_furiten: list[bool] = field(default_factory=lambda: [False] * 4)
    scores: list[int] = field(default_factory=lambda: [250] * 4)
    dealer: int = 0
    round_wind: int = 0
    honba: int = 0
    riichi_sticks: int = 0
    dora_indicators: list[str] = field(default_factory=list)
    furiten_discard_bases: list[set[int]] = field(default_factory=lambda: [set() for _ in range(4)])
    last_discard_actor: int | None = None
    last_discard_tile: str | None = None
    discard_history: list[tuple[int, str]] = field(default_factory=list)
    kyoku_index: int = -1

    @classmethod
    def from_init_payload(cls, kyoku_index: int, payload: dict) -> "GameState":
        """执行与函数名对应的业务逻辑。
        
        参数:
            cls (type[Self]): 当前类对象。
            kyoku_index (int): 相关索引值。
            payload (dict): 函数执行所需的输入参数。
        
        返回:
            'GameState': 函数执行结果。
        
        异常:
            无。
        """
        hands = [
            HandState.from_tiles(payload["hai0"]),
            HandState.from_tiles(payload["hai1"]),
            HandState.from_tiles(payload["hai2"]),
            HandState.from_tiles(payload["hai3"]),
        ]
        scores = [int(score) for score in str(payload.get("ten", "250,250,250,250")).split(",")]
        seed_parts = [int(part) for part in str(payload.get("seed", "")).split(",") if part]
        honba = seed_parts[1] if len(seed_parts) > 1 else 0
        riichi_sticks = seed_parts[2] if len(seed_parts) > 2 else 0
        round_wind = int(payload["round_wind"])
        dora_indicator = payload.get("dora_indicator")
        dora_indicators = [str(dora_indicator)] if dora_indicator is not None else []
        return cls(
            hands=hands,
            kyoku_index=kyoku_index,
            scores=scores,
            dealer=int(payload.get("oya", 0)),
            round_wind=round_wind,
            honba=honba,
            riichi_sticks=riichi_sticks,
            dora_indicators=dora_indicators,
        )

    def apply_draw(self, actor: int, tile: str) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
            tile (str): 精确牌字符串。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.temporary_furiten[actor] = False
        self.hands[actor].add_tile(tile)

    def apply_discard(self, actor: int, tile: str) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
            tile (str): 精确牌字符串。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.hands[actor].remove_tile(tile)
        self.discards[actor].append(tile)
        self.discard_history.append((actor, tile))
        self.furiten_discard_bases[actor].add(tile_to_base_index(tile))
        self.last_discard_actor = actor
        self.last_discard_tile = tile

    def apply_reach(self, actor: int, step: int) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
            step (int): 函数执行所需的输入参数。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        if step == 1:
            self.riichi_pending[actor] = True
        if step == 2 and self.scores[actor] >= 10:
            self.riichi_declared[actor] = True
            self.riichi_pending[actor] = False
            self.riichi_turns[actor] = len(self.discards[actor])
            self.scores[actor] -= 10

    def apply_meld(
        self,
        actor: int,
        kind: str,
        source: int,
        called_tile: str,
        consumed_tiles: list[str],
    ) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
            kind (str): 函数执行所需的输入参数。
            source (int): 动作来源玩家座位索引。
            called_tile (str): 被鸣牌或杠牌的精确牌字符串。
            consumed_tiles (list[str]): 副露消耗牌列表。
        
        返回:
            None: 无返回值。
        
        异常:
            ValueError: 当输入数据不满足函数约束时抛出。
        """
        if kind in {"chi", "pon", "daiminkan"}:
            if self.last_discard_actor != source:
                raise ValueError(
                    f"meld source mismatch: actor={actor} source={source} "
                    f"last_discard_actor={self.last_discard_actor}"
                )
            if self.last_discard_tile != called_tile:
                raise ValueError(
                    f"called tile mismatch: expected last discard {self.last_discard_tile}, got {called_tile}"
                )
            for tile in consumed_tiles:
                self.hands[actor].remove_tile(tile)
            self.melds[actor].append(
                MeldState(
                    kind=kind,
                    source=source,
                    called_tile=called_tile,
                    consumed_tiles=tuple(consumed_tiles),
                )
            )
            self.temporary_furiten[actor] = False
            self.last_discard_actor = None
            self.last_discard_tile = None
            return

        if kind == "ankan":
            for tile in consumed_tiles:
                self.hands[actor].remove_tile(tile)
            self.melds[actor].append(
                MeldState(
                    kind=kind,
                    source=source,
                    called_tile=called_tile,
                    consumed_tiles=tuple(consumed_tiles),
                )
            )
            return

        if kind == "kakan":
            self._upgrade_pon_to_kakan(actor=actor, called_tile=called_tile, consumed_tiles=consumed_tiles)
            return

        if kind == "nuki":
            return

        raise ValueError(f"unsupported meld kind: {kind}")

    def _upgrade_pon_to_kakan(self, actor: int, called_tile: str, consumed_tiles: list[str]) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
            called_tile (str): 被鸣牌或杠牌的精确牌字符串。
            consumed_tiles (list[str]): 副露消耗牌列表。
        
        返回:
            None: 无返回值。
        
        异常:
            ValueError: 当输入数据不满足函数约束时抛出。
        """
        if not consumed_tiles:
            raise ValueError("kakan is missing consumed tiles")

        added_tile = consumed_tiles[-1]
        self.hands[actor].remove_tile(added_tile)
        base_index = tile_to_base_index(called_tile)

        for meld_index, meld in enumerate(self.melds[actor]):
            if meld.kind == "pon" and meld.base_index == base_index:
                upgraded = MeldState(
                    kind="kakan",
                    source=meld.source,
                    called_tile=meld.called_tile,
                    consumed_tiles=meld.consumed_tiles + (added_tile,),
                )
                self.melds[actor][meld_index] = upgraded
                return

        raise ValueError(f"kakan without existing pon: actor={actor} tile={called_tile}")

    def has_open_melds(self, actor: int) -> bool:
        """判断与函数名对应的状态是否存在。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
        
        返回:
            bool: 判断结果。
        
        异常:
            无。
        """
        return any(meld.kind in {"chi", "pon", "daiminkan", "kakan"} for meld in self.melds[actor])

    def can_declare_riichi(self, actor: int) -> bool:
        """判断与函数名对应的操作是否可执行。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
        
        返回:
            bool: 判断结果。
        
        异常:
            无。
        """
        return (
            not self.riichi_declared[actor]
            and not self.riichi_pending[actor]
            and not self.has_open_melds(actor)
            and self.scores[actor] >= 10
        )

    def has_discard_furiten(self, actor: int, winning_tile_bases: set[int]) -> bool:
        """判断与函数名对应的状态是否存在。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
            winning_tile_bases (set[int]): 相关数据列表。
        
        返回:
            bool: 判断结果。
        
        异常:
            无。
        """
        return any(base_index in self.furiten_discard_bases[actor] for base_index in winning_tile_bases)

    def is_furiten(self, actor: int, winning_tile: str) -> bool:
        """判断与函数名对应的条件是否成立。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
            winning_tile (str): 函数执行所需的输入参数。
        
        返回:
            bool: 判断结果。
        
        异常:
            无。
        """
        winning_base = tile_to_base_index(winning_tile)
        return (
            self.has_discard_furiten(actor, {winning_base})
            or self.temporary_furiten[actor]
            or self.riichi_furiten[actor]
        )

    def mark_missed_ron(self, actor: int) -> None:
        """根据业务规则标记相关状态。
        
        参数:
            self (Self): 当前对象实例。
            actor (int): 玩家座位索引。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.temporary_furiten[actor] = True
        if self.riichi_declared[actor]:
            self.riichi_furiten[actor] = True

    def add_dora_indicator(self, tile: str) -> None:
        """向当前状态添加输入数据。
        
        参数:
            self (Self): 当前对象实例。
            tile (str): 精确牌字符串。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.dora_indicators.append(tile)

    def dora_base_indices(self) -> set[int]:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            set[int]: 集合结果。
        
        异常:
            无。
        """
        return {tile_to_base_index(dora_from_indicator(tile)) for tile in self.dora_indicators}
