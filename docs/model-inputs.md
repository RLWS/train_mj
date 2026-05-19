# 模型输入特征说明

更新时间：`2026-05-19`

本文记录当前 `turn` / `response` policy 模型实际吃到的输入张量、维度和每个维度的语义。这里描述的是 `build-dataset` 写入 `silver_samples` 后，训练和评估阶段从 parquet 读取的离线预编码特征。

## 1. 输入张量

每条样本包含一个局面向量和一组合法候选动作向量。模型只在合法候选集合内打分。

| 任务 | `state_feats` | `candidate_feats` | `candidate_mask` | `labels` |
|---|---:|---:|---:|---:|
| `turn` | `[B, 8173]` | `[B, C, 89]` | `[B, C]` | `[B]` |
| `response` | `[B, 8216]` | `[B, C, 91]` | `[B, C]` | `[B]` |

含义：

- `B` 是 batch size。
- `C` 是当前 batch 内最大候选数，受 config 中 `data.candidate_limit` 限制。
- `candidate_mask[b, c] = true` 表示该位置是真实合法候选；padding 候选会在 head 中被 mask 成 `-1e9`。
- `labels[b]` 是真实动作在 `legal_actions` 中的下标，而不是固定全局动作类别 id。

数据流：

```text
TurnSample/ResponseSample
-> encode_*_state + encode_*_candidate_actions
-> state_features_bin / candidate_features_bin
-> Manifest*Dataset
-> *BatchCollator
-> MahjongPolicyNet
```

## 2. 公共编码顺序

### 2.1 Exact tile order，37 维

所有 exact-tile one-hot / count block 都使用同一顺序：

```text
1m, 2m, 3m, 4m, 5m, 0m, 6m, 7m, 8m, 9m,
1p, 2p, 3p, 4p, 5p, 0p, 6p, 7p, 8p, 9p,
1s, 2s, 3s, 4s, 5s, 0s, 6s, 7s, 8s, 9s,
1z, 2z, 3z, 4z, 5z, 6z, 7z
```

`0m/0p/0s` 表示赤五，和普通 `5m/5p/5s` 分开编码。

### 2.2 Optional tile，38 维

前 37 维使用 exact tile order，最后 1 维表示 `missing / none`。

### 2.3 Actor，4 或 5 维

- 4 维 actor one-hot：`actor 0, 1, 2, 3`。
- 5 维 optional actor one-hot：`actor 0, 1, 2, 3, missing`。

## 3. `turn` state，8173 维

`TURN_STATE_DIM = 8173`。

| 范围 | 维度 | 语义 |
|---:|---:|---|
| `0..36` | 37 | 当前玩家手牌 exact tile counts |
| `37..184` | 148 | 四家弃牌河 exact tile counts，按玩家 `0,1,2,3` 各 37 维 |
| `185..222` | 38 | 最近一张弃牌，optional tile |
| `223..227` | 5 | 最近弃牌玩家，optional actor |
| `228..231` | 4 | 当前决策玩家 actor one-hot |
| `232` | 1 | 当前玩家是否已立直 |
| `233` | 1 | temporary furiten |
| `234` | 1 | riichi furiten |
| `235..239` | 5 | 自家副露计数，顺序为 `chi, pon, daiminkan, ankan, kakan` |
| `240..243` | 4 | 四家立直状态，按 actor `0,1,2,3` 排列 |
| `244..247` | 4 | 对手立直巡目，按 actor `0,1,2,3` 排列；当前玩家和未立直为 `0`，其他玩家为 `riichi_turn / 18` |
| `248..8119` | 7872 | 全局牌河顺序 token，最多 192 个弃牌事件；每个 token 为 `actor one-hot(4) + exact tile one-hot(37)`，不足部分为 0 |
| `8120..8123` | 4 | 庄家 actor one-hot |
| `8124..8127` | 4 | 自风 actor one-hot |
| `8128..8164` | 37 | dora base tile counts，按 exact tile order 的普通牌面槽位记录 |
| `8165..8168` | 4 | 四家点数，`score / 1000.0` |
| `8169` | 1 | 巡目，`turn_number / 18`，截断到 `1.0` |
| `8170` | 1 | 估计剩余摸牌数，`remaining_draws_estimate / 18`，截断到 `1.0` |
| `8171` | 1 | 本场数，`honba / 8`，截断到 `1.0` |
| `8172` | 1 | 立直棒数，`riichi_sticks / 8`，截断到 `1.0` |

## 4. `response` state，8216 维

`RESPONSE_STATE_DIM = TURN_STATE_DIM + 43 = 8216`。

前 `0..8172` 与 `turn` state 完全相同，后面追加响应窗口的目标牌信息：

| 范围 | 维度 | 语义 |
|---:|---:|---|
| `0..8172` | 8173 | 完整 `turn` state |
| `8173..8177` | 5 | `response_target.source`，optional actor |
| `8178..8215` | 38 | `response_target.tile`，optional tile |

## 5. `turn` candidate action，89 维

`TURN_ACTION_DIM = 89`。

| 范围 | 维度 | 语义 |
|---:|---:|---|
| `0..6` | 7 | 动作类型 one-hot |
| `7..43` | 37 | `action.tile` exact tile one-hot |
| `44..88` | 45 | discard-like 候选特征 |

动作类型顺序：

```text
discard, riichi_discard, tsumo, ankan, kakan, kyuushu_kyuuhai, unknown
```

对于 `tsumo / ankan / kakan / kyuushu_kyuuhai` 等非弃牌候选，只有动作类型和可选 tile one-hot 有效，后 45 维 discard-like 特征为 0。

### 5.1 turn discard-like 特征，45 维

以下索引是相对于 discard-like block 的局部索引；全局维度为 `44 + index`。

| index | 语义 |
|---:|---|
| `0` | 是否为可编码的 `discard / riichi_discard` 候选 |
| `1` | 当前向听数，`current_shanten / 6`，范围截断到 `[-1, 1]` |
| `2` | 弃牌后向听数，`after_shanten / 6`，范围截断到 `[-1, 1]` |
| `3` | 向听改善量，`current_shanten - after_shanten` |
| `4` | 弃牌后有效牌种类数，`len(effective_tiles) / 34` |
| `5` | 弃牌后有效牌枚数，`effective_tile_copies / 136` |
| `6` | 弃牌后完整面子摘要，`complete_sets / 4` |
| `7` | 弃牌后对子数，`pairs / 7` |
| `8` | 弃牌后两面搭子数，`ryanmen / 8` |
| `9` | 弃牌后嵌张/边张搭子数，`closed_waits / 8` |
| `10` | 弃牌后孤张数，`isolated / 14` |
| `11` | 弃牌后同 base tile 剩余张数，`after_counts[base] / 4` |
| `12` | 被切牌是否幺九牌 |
| `13` | 被切牌是否字牌 |
| `14` | 被切牌 base 是否五牌或 dora，实际值为 `min(1, is_five_base + is_dora)` |
| `15` | 被切牌是否 exact red tile，即 `0m/0p/0s` |
| `16` | 当前玩家是否已立直 |
| `17` | 当前候选是否 `riichi_discard` |
| `18` | 被切牌是否为其他玩家现物 |
| `19` | 被切牌可见枚数，`visible_count / 4` |
| `20` | 是否满足当前简化筋安全判断 |
| `21` | 是否满足当前简化 one-chance 判断 |
| `22` | 弃牌后是否听牌，`after_shanten == 0` |
| `23` | 弃牌后是否进向听，`after_shanten < current_shanten` |
| `24` | 弃牌后是否退向听，`after_shanten > current_shanten` |
| `25` | 被切牌在自家牌河出现次数，`own_count / 4` |
| `26` | 被切牌在其他三家牌河出现总次数，`other_count / 4` |
| `27` | 有几个对手打过这张牌，`other_players / 3` |
| `28` | 有几个对手最近 6 巡打过这张牌，`recent_other_players / 3` |
| `29` | 有几个对手前 6 巡打过这张牌，`early_other_players / 3` |
| `30` | 这张牌最早被对手打出的对手牌河巡目，`first_turn / 18`；未出现为 `0` |
| `31` | 这张牌最近被对手打出的对手牌河巡目，`last_turn / 18`；未出现为 `0` |
| `32` | 是否为三家现物 |
| `33` | 是否至少两家现物 |
| `34` | 是否至少一家最近 6 巡现物 |
| `35` | 是否为下家现物，相对当前 actor 的 `(actor + 1) % 4` |
| `36` | 是否为对面现物，相对当前 actor 的 `(actor + 2) % 4` |
| `37` | 是否为上家现物，相对当前 actor 的 `(actor + 3) % 4` |
| `38` | 是否为下家最近 6 巡现物 |
| `39` | 是否为对面最近 6 巡现物 |
| `40` | 是否为上家最近 6 巡现物 |
| `41` | 是否为 actor 0 的现物；当前玩家自己固定为 `0` |
| `42` | 是否为 actor 1 的现物；当前玩家自己固定为 `0` |
| `43` | 是否为 actor 2 的现物；当前玩家自己固定为 `0` |
| `44` | 是否为 actor 3 的现物；当前玩家自己固定为 `0` |

## 6. `response` candidate action，91 维

`RESPONSE_ACTION_DIM = 91`。

| 范围 | 维度 | 语义 |
|---:|---:|---|
| `0..6` | 7 | 动作类型 one-hot |
| `7..43` | 37 | `action.tile` exact tile one-hot |
| `44..47` | 4 | `action.source` actor one-hot；没有 source 时全 0 |
| `48..90` | 43 | response call/context 候选特征 |

动作类型顺序：

```text
pass, chi, pon, daiminkan, ron, chankan, unknown
```

### 6.1 response call/context 特征，43 维

以下索引是相对于 call/context block 的局部索引；全局维度为 `48 + index`。

| index | 语义 |
|---:|---|
| `0` | 候选是否 call 类动作：`chi / pon / daiminkan` |
| `1` | call 是否可应用，即 consumed tiles 在当前手牌中可扣除 |
| `2` | 当前向听数，`current_shanten / 6`，范围截断到 `[-1, 1]` |
| `3` | call 后标准形向听数，按固定副露数计算，`after_shanten / 6` |
| `4` | 向听改善量，`current_shanten - after_shanten` |
| `5` | 当前有效牌种类数，`len(current_effective_tiles) / 34` |
| `6` | call 后有效牌种类数，`len(after_effective_tiles) / 34` |
| `7` | 有效牌种类变化，`(after - current) / 34`，范围截断到 `[-1, 1]` |
| `8` | 当前有效牌枚数，`current_effective_tile_copies / 136` |
| `9` | call 后有效牌枚数，`after_effective_tile_copies / 136` |
| `10` | 有效牌枚数变化，`(after - current) / 136`，范围截断到 `[-1, 1]` |
| `11` | call 是否消耗 exact red tile |
| `12` | call 是否消耗 dora 或 exact red tile |
| `13` | 目标牌是否 dora |
| `14` | call 是否消耗对子 |
| `15` | call 是否拆搭子 |
| `16` | `pon` 目标是否役牌 |
| `17` | 目标牌是否幺九牌 |
| `18` | 目标牌是否字牌 |
| `19` | `chi` 形状是否两面 |
| `20` | `chi` 形状是否嵌张 |
| `21` | `chi` 形状是否边张 |
| `22` | call 后完整面子摘要，`complete_sets / 4` |
| `23` | call 后对子数，`pairs / 7` |
| `24` | call 后两面搭子数，`ryanmen / 8` |
| `25` | call 后嵌张/边张搭子数，`closed_waits / 8` |
| `26` | call 后孤张数，`isolated / 14` |
| `27` | 巡目，`turn_number / 18`，截断到 `1.0` |
| `28` | 估计剩余摸牌数，`remaining_draws_estimate / 18`，截断到 `1.0` |
| `29` | 当前有效牌枚数，按手牌、牌河、四家副露、宝牌指示牌等可见牌修正后除以 `136` |
| `30` | call 后有效牌枚数，按可见牌和本次消耗牌修正后除以 `136` |
| `31` | 可见牌修正后的有效牌枚数变化，`(after - current) / 136`，范围截断到 `[-1, 1]` |
| `32` | 本次 call 是否会从门清变为副露 |
| `33` | call 后手牌是否处于副露状态 |
| `34` | call 后是否有明显役牌路线，包括已有/本次役牌刻子，或手内役牌对子候选 |
| `35` | call 后是否仍可能断幺路线 |
| `36` | call 后是否有明显染手路线，手牌与副露的数牌只落在一种花色 |
| `37` | call 后是否有明显对对和路线，粗略使用刻子副露、暗刻和对子数量判断 |
| `38` | call 后是否没有明显开副露役路，即无役风险启发式 |
| `39` | 当前七对子向听数，`chiitoitsu_shanten / 6`，范围截断到 `[-1, 1]` |
| `40` | 当前七对子路线是否活跃且本次 call 会破坏门清七对子路线 |
| `41` | 当前国士向听数，`kokushi_shanten / 6`，范围截断到 `[-1, 1]` |
| `42` | 当前国士路线活跃或特殊闭门路线明显优于 call 后标准形时置位 |

对 `pass / ron / chankan`，call 专属标志一般为 0；但当前向听、有效牌、目标牌属性、形状摘要和巡目信息仍会被填充，用于让模型在同一候选集合内比较 `pass/ron/chankan` 和 call 候选。

新增的开副露役路特征是启发式信号，不等价于完整番种判定。它们用于降低“鸣牌后无役”“七对子/国士等门清路线被错误鸣牌破坏”这类 response false call。

## 7. 模型如何消费这些输入

`MahjongPolicyNet` 不直接看到原始 JSON。它只看到：

```text
state_feats: [B, state_dim]
candidate_feats: [B, C, action_dim]
candidate_mask: [B, C]
```

处理方式：

1. `MahjongEncoder.state_encoder` 把 `state_feats` 编码为 `[B, H]`。
2. `MahjongEncoder.action_encoder` 把 `candidate_feats` 编码为 `[B, C, H]`。
3. `TurnHead` / `response_head` 将 state hidden broadcast 到每个候选，与 candidate hidden 拼接后逐候选打分。
4. padding 候选通过 `candidate_mask` mask 掉。
5. loss 是合法候选集合内的交叉熵，label 是真实动作在候选列表中的位置。

当前 `turn` 和 `response` 是单任务训练入口；模型类有共享 encoder + 两个 head 的接口，但还没有实现真正 `task: multitask` 的 batch 混合与 loss 聚合。

## 8. 数据重建提醒

`state_features_bin` 和 `candidate_features_bin` 是固定维度 float32 二进制。只要上述任何维度或语义发生变化，都必须重新运行：

```bash
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 32 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
```

否则训练/评估阶段会出现 feature byte length mismatch，或者更隐蔽的旧语义污染。

2026-05-09 起，`response` 的 `candidate_features` 从 `77` 维增加到 `91` 维。

2026-05-19 起，`turn` 的 `candidate_features` 从 `85` 维增加到 `89` 维，新增按绝对座位拆开的现物特征。`turn` / `response` 的 `state_features` 也同步加入四家立直状态、对手立直巡目和最多 192 个全局牌河顺序 token。已有 `data/silver_samples`、manifest 和旧 checkpoint 不再适用于当前代码，下一轮训练前必须重建数据并重训。
