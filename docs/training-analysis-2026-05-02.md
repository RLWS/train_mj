# Turn / Response 训练分析记录

更新时间：`2026-05-02`

本文记录当前 `outputs/checkpoints/turn_local_smoke` 与 `outputs/checkpoints/response_local_smoke` 两次训练的结果分析，以及下一轮使用 20 万局数据和 8 张 P800 的改进方向。

## 1. 数据与实验背景

当前本地数据规模来自 `4233` 局牌谱：

- `turn` 样本：`631016`
- `response` 样本：`1876854`
- train / valid / test 按局切分：
  - `train_turn=500969`，`valid_turn=64229`，`test_turn=65818`
  - `train_response=1490034`，`valid_response=191082`，`test_response=195738`

动作分布极不均衡：

| 任务 | 主类 | 主类占比 | 非主类占比 |
|---|---:|---:|---:|
| turn | `discard` | `97.3871%` | `2.6129%` |
| response | `pass` | `97.9326%` | `2.0674%` |

`response` 比 `turn` 更不均衡一些，但差距不是数量级变化。`response` 准确率异常高的更关键原因是候选集合结构：valid 中 `172909 / 191082 = 90.49%` 的 response 样本只有 `pass` 一个候选，模型不需要判断也会天然正确。

## 2. Turn 训练结论

配置：

- checkpoint 目录：`outputs/checkpoints/turn_local_smoke`
- 训练配置：`hidden_size=256`，`batch_size=1024`，`epochs=28`
- 类别权重：`discard=1`，`riichi_discard=4`，`tsumo=6`，`ankan/kakan/kyuushu=8`

`best.pt` 对应 valid epoch 11：

| 指标 | 数值 |
|---|---:|
| valid loss | `1.0986` |
| top1 / accuracy | `0.6318` |
| top3 | `0.9088` |
| top5 | `0.9682` |
| macro top1 | `0.7662` |

按类别看：

| label | samples | top1 |
|---|---:|---:|
| discard | `62590` | `0.6266` |
| riichi_discard | `1059` | `0.7819` |
| tsumo | `472` | `1.0000` |
| ankan | `54` | `0.3889` |
| kakan | `45` | `0.8000` |
| kyuushu_kyuuhai | `9` | `1.0000` |

训练曲线显示明显过拟合：

- train loss 从 `1.2348` 降到 `0.7535`
- train accuracy 从 `0.5646` 升到 `0.7224`
- valid loss 在 epoch 6 最好：`1.0573`
- valid accuracy 在 epoch 11 最好：`0.6318`
- valid macro accuracy 在 epoch 4 最好：`0.8831`
- epoch 28 valid loss 已升到 `1.2279`，accuracy 降到 `0.6135`

因此当前 `28` epoch 对 4k 局数据过长。后续应按 valid loss / macro / non-majority 指标早停，而不是固定训练到 28 epoch。

### Turn 错误结构

turn 的主要错误不是动作大类错误，而是同为 `discard` 时选错具体牌。

valid 中 label 为 `discard` 的样本有 `62590` 条，模型预测成 `discard` 的有 `61613` 条，说明大多数时候动作类型是对的。但 exact action top1 只有 `62.66%`，而 top3 达到 `90.71%`。这说明模型已经能把合理候选排到前几名，但缺少把 top2/top3 拉到 top1 的信息。

下一轮 turn 的主要提升点不是先堆复杂模型，而是补充能区分 discard 候选优劣的上下文特征。

## 3. Response 训练结论

配置：

- checkpoint 目录：`outputs/checkpoints/response_local_smoke`
- 训练配置：`hidden_size=128`，`batch_size=1024`，`epochs=1`
- 类别权重：`pass=1`，`chi/pon/daiminkan=8`，`ron/chankan=12`

`best.pt` 对应 valid epoch 1：

| 指标 | 数值 |
|---|---:|
| valid loss | `0.1095` |
| top1 / accuracy | `0.9752` |
| top3 | `0.9995` |
| top5 | `1.0000` |
| macro top1 | `0.5792` |

按类别看：

| label | samples | top1 |
|---|---:|---:|
| pass | `187089` | `0.9782` |
| pon | `1765` | `0.8929` |
| chi | `1199` | `0.6038` |
| ron | `1019` | `1.0000` |
| daiminkan | `9` | `0.0000` |
| chankan | `1` | `0.0000` |

`response` 的 overall accuracy 一开始就高，主要不是模型能力强，而是评估被两个因素支配：

1. `pass` 标签占 `97.91%`。
2. valid 中 `90.49%` 的样本只有一个合法候选，也就是只有 `pass`。

“永远预测 pass”的朴素基线在这个 valid split 上约为 `97.91%`，甚至高于当前模型的 `97.52%` top1。因此 response 的 overall accuracy 不能作为有效模型质量判断。

但模型并不是完全没学到东西：在非 pass 样本上，当前模型 top1 约为 `83.12%`，其中 `pon=89.29%`、`ron=100%`、`chi=60.38%`。这说明类别权重让模型开始学习副露/荣和，但也带来了 false call 风险。

### Response 混淆结构

valid argmax 混淆统计：

| label -> pred | count |
|---|---:|
| pass -> pass | `183015` |
| pass -> chi | `2047` |
| pass -> pon | `1960` |
| pass -> ron | `60` |
| pass -> daiminkan | `7` |
| chi -> chi | `724` |
| chi -> pass | `458` |
| pon -> pon | `1576` |
| pon -> pass | `189` |
| ron -> ron | `1019` |
| daiminkan -> pon | `8` |
| chankan -> ron | `1` |

当前主要错误有两类：

- false positive call：本来应该 pass，却预测 chi/pon，合计超过 `4000` 条。
- false negative call：本来应该 chi/pon，却预测 pass，尤其 `chi -> pass = 458`。

这与类别权重有关。`response` 的非 pass 权重比 pass 高 `8-12` 倍，确实会推动模型更多尝试 call。但高 overall accuracy 不是权重带来的，而是 `pass` 标签和单候选样本占比过高带来的。

## 4. 下一轮优先级

### 4.1 先修 response 评估口径

下一轮必须新增 response 专用指标，否则会继续被 `pass` 掩盖：

- `candidate_count > 1` 子集的 top1 / macro top1
- 排除单候选 pass-only 样本后的 valid accuracy
- `pass` vs `non_pass` 二分类 precision / recall / F1
- false call rate：`pass -> chi/pon/ron/...`
- missed call rate：`chi/pon/ron/... -> pass`
- 按候选类型统计：只有 `pass+pon`、`pass+chi`、`pass+ron`、多候选混合

`response` 的 checkpoint 选择也不应再只看 overall accuracy，应优先看 non-pass F1、false call rate 和 candidate_count>1 子集表现。

### 4.2 Response 训练数据要过滤或重采样

单候选 `pass` 样本没有排序学习价值，只会放大 accuracy 并消耗训练时间。建议：

1. 训练时默认过滤 `candidate_count == 1` 的 response 样本，或把它们权重降到接近 0。
2. valid/test 报告中同时保留全量指标和 `candidate_count > 1` 指标。
3. 对 `pass` hard negative 采样：只保留有 call/ron 机会但实际选择 pass 的样本。
4. 对 `chi/pon/ron` 做分层采样，保证每个 batch 有足够非 pass。

这样 response 的学习目标会从“识别绝大多数 trivially pass”变成“在确实有可响应动作时，判断该不该响应”。

### 4.3 特征优先于复杂模型

当前 turn top3 很高、response top3 接近满分，说明模型已经能粗排，但缺少细粒度决策信息。下一轮优先补特征：

Turn：

- 牌河顺序，而不只是 discard counts
- 各家立直状态、立直巡目、现物/筋/one-chance 按对手拆分
- 所有玩家副露信息，而不只是 own_melds
- 摸切/手出标记
- 候选 discard 的 dora/赤宝牌损失、向听变化、有效牌变化
- 点差、亲家、局面阶段、剩余巡目更细粒度特征

Response：

- call 后向听数变化、有效牌变化、形状变化
- chi/pon 后是否有役、是否断幺/役牌/染手趋势
- call 是否消耗 dora/赤牌/关键搭子
- 对手立直/副露下的防守风险
- 被打出的牌是否危险、是否现物、是否高价值
- source 玩家、座位关系、巡目、点差、亲家状态

`ron` 当前接近满分，多半因为合法动作本身已经由规则层判断了和牌机会。response 的难点主要是 `chi/pon` 是否应该叫，以及 pass hard negative。

### 4.4 20 万局 + 8 张 P800 的建议路线

不要先上复杂模型。更稳妥的路线是：

1. 先在 20 万局上重建 `silver_samples` 和 manifest。
2. 先跑当前模型的大数据 baseline，确认大数据是否解决过拟合和稀有类波动。
3. 训练工程先补齐真正的 8 卡 DDP、bf16 autocast、grad accumulation、step eval/save。
4. response 先过滤或降权 pass-only 样本，再训练。
5. 指标体系补齐后，再做特征增强实验。
6. 最后再尝试更复杂模型。

复杂模型适合放在第三阶段：

- hidden size 从 `256/512` 做对照
- 加 history encoder 或 tile sequence encoder
- turn/response 共享 encoder，多头训练
- Transformer/attention 用于牌河与副露时序

在当前阶段，直接换复杂模型的风险是：response accuracy 仍被 pass-only 样本掩盖，turn 仍会因为上下文不足而在 top2/top3 之间摇摆，难以判断模型结构是否真的有贡献。

## 5. 建议的下一组实验

### 实验 A：当前模型 + 20 万局

- turn：保留当前候选排序模型，`hidden_size=256/512` 各跑一组
- response：过滤或低权重 `candidate_count == 1`
- 训练：DDP 8 卡，bf16，global batch 先从 `8192-16384` 试
- epoch：先跑 `1-3`，每半个 epoch 或固定 step 做 valid
- 保存：`best_loss`、`best_accuracy`、`best_macro`、`best_non_pass_f1`

### 实验 B：response 训练目标改造

两种方向二选一：

- 继续 listwise ranking，但只训练 `candidate_count > 1` 或 hard-negative pass。
- 改成两阶段：先 `respond vs pass` 二分类，再在非 pass 内做 `chi/pon/ron/...` 排序。

当前数据形态下，两阶段 response 可能更容易解释和调权重。

### 实验 C：特征增强

先不改模型结构，只加离线预编码特征。每次只加一组特征，保留同样训练配置，观察：

- turn exact top1 是否提升
- turn top3 是否变化不大但 top1 上升
- response candidate_count>1 top1 / non-pass F1 是否提升
- false call rate 是否下降

如果这些指标有明确收益，再考虑更深模型吸收更复杂时序信息。
