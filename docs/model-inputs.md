# 模型输入特征说明

更新时间：`2026-10-27`

当前实验分支使用 `full_state_v2`，旧 checkpoint / bronze / silver 不能复用；完整操作见 [full-state-training.md](full-state-training.md)。

本文记录当前 `turn` / `response` policy 模型实际吃到的输入张量、维度和每个维度的语义。这里描述的是 `build-dataset` 写入 `silver_samples` 后，训练和评估阶段从 parquet 读取的离线预编码特征。

## 1. 输入张量

每条样本包含一个局面向量和一组合法候选动作向量。模型只在合法候选集合内打分。

| 任务 | `state_feats` | `candidate_feats` | `candidate_mask` | `labels` |
|---|---:|---:|---:|---:|
| `turn` | `[B, 11969]` | `[B, C, 89]` | `[B, C]` | `[B]` |
| `response` | `[B, 12012]` | `[B, C, 91]` | `[B, C]` | `[B]` |

含义：

- `B` 是 batch size。
- `C` 是当前 batch 内最大候选数，受 config 中 `data.candidate_limit` 限制。
- `candidate_mask[b, c] = true` 表示该位置是真实合法候选；padding 候选会在 head 中被 mask 成 `-1e9`。
- `labels[b]` 是真实动作在 `legal_actions` 中的下标，而不是固定全局动作类别 id。

数据流：

```text
TurnSample（decision_type=turn|response）
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

### 2.3 座位与点数约定

原始 JSON 保留平台绝对座位。编码器同时旋转 state 和 candidate source：
`0=自家、1=下家、2=对面、3=上家`。所有四家数组、dealer、历史 actor、
副露 source 和 response target 同步旋转；`seat_wind/round_wind` 是风位，不能旋转。
返回的候选动作仍使用调用方绝对座位，不改变候选顺序。

`scores` 使用实际点数，例如 25000；GameState 内部仍使用天凤百点单位，
只在 snapshot 边界乘 100。state 中点数除以 1000，分差除以 10000。

## 3. turn state，11969 维

| 范围 | 维度 | 语义 |
|---|---:|---|
| 0..247 | 248 | 原有手牌、四家牌河、最近弃牌、actor、振听、副露类型计数、立直状态及巡目；座位改为相对座位 |
| 248..8695 | 8448 | 192 个弃牌 token，每个 44 维 |
| 8696..8699 | 4 | 相对庄家 one-hot |
| 8700..8703 | 4 | 自风 one-hot |
| 8704..8740 | 37 | 宝牌计数，保留重复指示牌带来的倍数 |
| 8741..8744 | 4 | 相对座位点数 / 1000 |
| 8745..10120 | 1376 | 四家各 4 个副露槽，每槽 86 维 |
| 10121..11944 | 1824 | 最多 32 个非弃牌公开事件，每个 57 维 |
| 11945..11948 | 4 | 场风 one-hot |
| 11949..11952 | 4 | 场内局号 one-hot，0..3 对应一至四局；不使用含连庄的回放序号 |
| 11953..11955 | 3 | 东风 / 半庄 / 未知赛制 |
| 11956..11957 | 2 | 是否为计划末局、是否已进入延长场；不预测实际何时终局 |
| 11958..11959 | 2 | 剩余摸牌数已知标志、remaining_tiles / 70 |
| 11960..11963 | 4 | 各家与自家的点差 / 10000 |
| 11964 | 1 | 非弃牌公开事件数量 / 32 |
| 11965..11968 | 4 | 自家实际摸牌次数 / 18、remaining_tiles / 72、本场 / 8、立直棒 / 8；均上截断为 1 |

弃牌 token：相对 actor(4) + exact tile(37) + 摸切信息已知(1) + 摸切(1) + 立直宣言牌(1)。
摸切由天凤 136 牌 ID 与本家最近摸牌 ID 比较得到，不再依赖 XML 标签大小写。
未知摸切使用 null，不能当成手切。历史不移除被鸣走的弃牌。

副露槽：存在标志(1) + kind(5: chi/pon/daiminkan/ankan/kakan) + source(5，含缺失) +
called_tile(38，含缺失) + consumed_tiles exact counts(37)。
暗杠 consumed 包含四张；加杠保存升级后的副露，保留原碰来源。

公开事件：kind(8: reach1/reach2/chi/pon/daiminkan/ankan/kakan/dora) +
actor(5，含缺失) + source(5，含缺失) + tile(38，含缺失) + discard_count/192(1)。
顺序按发生顺序保存，discard_count 标记事件发生前已有多少次弃牌，用于和牌河合并排序。
不包含对手摸牌内容、隐藏手牌或终局结果。超过容量时报错，不静默截断。
turn 立直样本在记入自身 reach1 之前截取状态，防止把标签暴露给模型。

剩余摸牌机会初始为 70，每次普通摸牌或岭上摸牌递减一次；不在宣杠时再次递减，
避免将岭上补牌及补充王牌重复计数。这是四麻计数，不是完整牌山模拟。

## 4. response state，12012 维

前 11969 维与 turn 相同；追加 response_target 的相对 source(5，含缺失) 和 tile(38，含缺失)。

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
| `41` | 是否为相对 actor 0 的现物；当前玩家自己固定为 `0` |
| `42` | 是否为相对 actor 1 的现物；当前玩家自己固定为 `0` |
| `43` | 是否为相对 actor 2 的现物；当前玩家自己固定为 `0` |
| `44` | 是否为相对 actor 3 的现物；当前玩家自己固定为 `0` |

## 6. `response` candidate action，91 维

`RESPONSE_ACTION_DIM = 91`。

| 范围 | 维度 | 语义 |
|---:|---:|---|
| `0..6` | 7 | 动作类型 one-hot |
| `7..43` | 37 | `action.tile` exact tile one-hot |
| `44..47` | 4 | `action.source` 相对 actor one-hot；没有 source 时全 0 |
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

2026-06-01 起，样本状态会从 Tenhou `INIT seed` 写入 `round_wind`。该字段不增加 state 向量维度，但会影响 response candidate 的役牌路线判断，尤其是 `pon` 目标是否场风牌、`features[34]` 的役牌路线和 `features[38]` 的无明显开副露役路风险。旧 `bronze_events` 没有 `round_wind`，不再兼容；需要从 `preprocess` 开始重建数据，避免南场/西场被错误当作东场。

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
3. turn/response 都复用 `TurnHead` 这一个候选打分类；它将 state hidden broadcast 到每个候选，与 candidate hidden 拼接后逐候选打分。
4. padding 候选通过 `candidate_mask` mask 掉。
5. loss 是合法候选集合内的交叉熵，label 是真实动作在候选列表中的位置。

当前 `turn` 和 `response` 均支持单任务训练入口；多任务训练使用 `MultitaskMahjongPolicyNet`，保留任务专属 state/action projection 与 head，并共享状态主干。P800 主配置仍默认使用 v1.1 epoch-sequential multitask，v1.2 batch-interleave 作为显式实验配置保留。

## 8. 数据重建提醒

**full_state_v2 必须从原始 XML 重新 preprocess，之后重建 silver 和 manifest 并从头训练。**
旧 bronze 缺少 schema 标记会被 build-dataset 拒绝；旧 checkpoint 会被 eval/serving 明确拒绝。
以下日期记录是历史变化，不替代本分支迁移要求。

`state_features_bin` 和 `candidate_features_bin` 是固定维度 float32 二进制。只要上述任何维度或语义发生变化，都必须重新运行：

```bash
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 32 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
```

否则训练/评估阶段会出现 feature byte length mismatch，或者更隐蔽的旧语义污染。

2026-05-09 起，`response` 的 `candidate_features` 从 `77` 维增加到 `91` 维。

2026-05-19 起，`turn` 的 `candidate_features` 从 `85` 维增加到 `89` 维，新增按座位拆开的现物特征（v2 编码时改为相对座位）。`turn` / `response` 的 `state_features` 也同步加入四家立直状态、对手立直巡目和最多 192 个全局牌河顺序 token。已有 `data/silver_samples`、manifest 和旧 checkpoint 不再适用于当前代码，下一轮训练前必须重建数据并重训。

2026-06-01 起，`round_wind` 会进入样本状态并修正 response 无役风险/役牌路线启发式。虽然 `state_features_bin` / `candidate_features_bin` 维度不变，但 response candidate 语义发生变化；要获得正确的南场/西场役牌判断，需要重新运行 `preprocess`、`build-dataset` 和 `make-manifest`。

2026-06-22 起，turn discard candidate feature 仍保持 `TURN_ACTION_DIM=89`，但语义已经变化：向听与有效牌计算会把 `own_melds` 视为固定面子；可见牌与有效等待计数会包含四家副露里的牌，包括可能已从 `all_discards` 移除的被鸣牌。重新训练 turn 前必须重建 `silver_samples` 和 manifest。
