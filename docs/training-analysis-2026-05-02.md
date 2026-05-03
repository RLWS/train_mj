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

## 6. 2026-05-03 已落地的 response 优化

本轮先按第 4.1 / 4.2 节的方向改训练与评估口径，没有改 turn 的数据、loss、head 或配置。

已实现：

- `configs/response_local_smoke.yaml` 与 `configs/response_p800.yaml` 增加 `train.response_min_train_candidate_count: 2`，response 训练默认过滤 `candidate_count == 1` 的 pass-only 样本。
- valid/test 评估仍保留全量样本，不过滤 pass-only；报告新增 `response` 专用诊断：
  - `single_candidate` 与 `multi_candidate` 的 top1/top3/top5 和 macro top-k。
  - `pass` vs `non_pass` 的 precision / recall / F1。
  - `false_call_rate` 与 `missed_call_rate`。
  - `candidate_sets`，按 `pass_only`、`pass+chi`、`pass+pon`、`pass+ron` 等候选集合聚合。
  - `confusion`，记录 `label->pred`，例如 `pass->chi`、`chi->pass`。
- response 训练期间也会把同一组诊断写入 `train_metrics.json` / `epoch_metrics.jsonl`。
- response 的 `best.pt` / `best_response.pt` 改为按 `non_pass_f1 - false_call_rate` 选择；`best_accuracy.pt`、`best_loss.pt`、`best_macro.pt` 仍保留，便于横向比较。

对 turn 的影响边界：

- 当前 `MahjongPolicyNet` 仍是共享 encoder + `turn_head` / `response_head` 的多头接口，但训练入口仍只支持单任务 `turn` 或 `response`。也就是说，训练 turn 时实例化的是 turn 维度的 encoder 和 turn head；训练 response 时实例化的是 response 维度的 encoder 和 response head。
- 本轮 response 过滤只在 `_build_dataset(task=response, split=train)` 生效；`turn` 数据集仍使用 `ManifestTurnDataset`，`turn_local_smoke.yaml` / `turn_p800.yaml` 没有变化。
- eval 报告中只有 response checkpoint 会填充 `response` 专用诊断；turn checkpoint 的该字段为 `null`。

后续如果要回到最初规划里的真正多任务训练，需要单独实现 `task: multitask` 的 batch 混合、head 路由、loss 聚合和不同状态/动作维度的共享编码处理。本轮没有做这一步，避免 response 调参与 turn baseline 相互污染。

## 7. 2026-05-03 response 重训结果

本轮重训配置：

- checkpoint 目录：`outputs/checkpoints/response_local_smoke`
- 训练配置：`hidden_size=128`，`batch_size=1024`，`epochs=10`
- 训练过滤：`response_min_train_candidate_count=2`
- response 类别权重：`pass=1`，`chi/pon/daiminkan=8`，`ron/chankan=12`
- 训练集从 `1490034` 条 response 样本过滤到 `141423` 条，过滤 `1348611` 条 pass-only 单候选样本。
- valid/test 评估仍使用全量样本。

### 7.1 与旧 response baseline 对比

旧 baseline 是 2026-05-02 的 `response_local_smoke` 结果，只训练 `1` epoch，且训练集未过滤 pass-only 样本。新结果使用 2026-05-03 的 `best.pt`。

| 指标 | 旧 valid | 新 valid | 新 test |
|---|---:|---:|---:|
| overall top1 | `0.9752` | `0.9788` | `0.9793` |
| macro top1 | `0.5792` | `0.7515` | `0.7597` |
| multi-candidate top1 | 未记录 | `0.7766` | `0.7804` |
| multi-candidate macro top1 | 未记录 | `0.7162` | `0.7248` |
| non-pass precision | 未记录 | `0.4962` | `0.5040` |
| non-pass recall | 约 `0.8312` | `0.8034` | `0.8114` |
| non-pass F1 | 未记录 | `0.6135` | `0.6218` |
| false call rate | 约 `0.0218` | `0.0174` | `0.0170` |
| missed call rate | 约 `0.1688` | `0.1966` | `0.1886` |
| chi top1 | `0.6038` | `0.5822` | `0.5778` |
| pon top1 | `0.8929` | `0.8334` | `0.8434` |
| ron top1 | `1.0000` | `1.0000` | `1.0000` |

结论：

- 新训练显著提高了 `macro top1` 和 `candidate_count > 1` 子集表现，说明过滤 pass-only 样本后，模型确实更专注于有响应选择空间的样本。
- false call 明显下降：valid 从旧模型约 `pass->call = 4074 / 187089 = 2.18%` 降到 `3257 / 187089 = 1.74%`；test 为 `1.70%`。
- 代价是漏叫上升：valid 的 `missed_call_rate=19.66%`，旧模型约为 `16.88%`。也就是说新模型更保守，少误叫但多漏掉一部分 `chi/pon`。
- valid 与 test 指标非常接近，说明这轮改善不是明显 valid 过拟合。

### 7.2 epoch 选择

按不同标准选择的 epoch 不一致：

| 选择标准 | epoch | valid loss | multi top1 | multi macro top1 | non-pass F1 | false call | missed call | chi | pon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best loss | `3` | `0.1062` | `0.6764` | `0.5486` | `0.5516` | `0.0292` | `0.0987` | `0.6380` | `0.9569` |
| best macro | `7` | `0.1156` | `0.7292` | `0.7246` | `0.5811` | `0.0230` | `0.1493` | `0.5721` | `0.8941` |
| best response score | `10` | `0.1444` | `0.7766` | `0.7162` | `0.6135` | `0.0174` | `0.1966` | `0.5196` | `0.8334` |

当前 `best.pt` / `best_response.pt` 选择的是 epoch 10，因为现有 response score 是：

```text
non_pass_f1 - false_call_rate
```

这个选择偏向减少 false call，并提高 pass/non-pass 二分类 F1。它适合保守 response 策略，但会牺牲 `chi/pon` recall。若下一轮目标是更积极鸣牌，应把 checkpoint selection 改为带 recall 下限或分类型加权的分数。

### 7.3 错误结构

valid confusion：

| label -> pred | count |
|---|---:|
| `pass -> chi` | `1818` |
| `pass -> pon` | `1362` |
| `pass -> ron` | `60` |
| `pass -> daiminkan` | `17` |
| `chi -> pass` | `491` |
| `pon -> pass` | `293` |
| `chi -> chi` | `698` |
| `pon -> pon` | `1471` |
| `ron -> ron` | `1019` |

test confusion 与 valid 基本一致：

| label -> pred | count |
|---|---:|
| `pass -> chi` | `1792` |
| `pass -> pon` | `1381` |
| `pass -> ron` | `64` |
| `chi -> pass` | `485` |
| `pon -> pass` | `283` |
| `ron -> ron` | `1030` |

剩余主问题仍是两类：

- false call：真实 `pass` 但模型预测 `chi/pon`，valid 共 `3257` 条。
- missed call：真实 `chi/pon` 但模型预测 `pass`，valid 共 `785` 条，其中 `chi -> pass = 491`、`pon -> pass = 293`。

`ron` 继续稳定接近满分，说明当前 response 的核心难点仍是 `chi/pon` 是否值得叫，而不是规则层已确认的荣和机会。

### 7.4 需要注意的评估细节

`train_metrics.json` 里的 per-kind accuracy 使用 argmax exact index；`eval` 报告里的 top1 使用 label rank，允许并列分数下 label 排名为 1。因此新 valid 的 `chi` 在训练日志里是 `0.5196`，在 eval 报告里是 `0.5822`。

这个差异提示当前 response action 特征仍不够细，多个同类候选可能拿到相同或近似分数。尤其 `chi` 有不同 consumed tiles / pattern，`pon` 有赤牌消耗组合差异；如果 action encoder 没有充分表达这些细节，模型即使判断“应该 chi/pon”，也可能无法稳定选中精确 label action。

### 7.5 下一轮建议

优先级从高到低：

1. 调整 response checkpoint selection，不只用 `non_pass_f1 - false_call_rate`。建议试：
   - `non_pass_f1 - 0.5 * false_call_rate`
   - `0.5 * non_pass_f1 + 0.3 * multi_candidate.macro_top1 + 0.2 * chi_accuracy - false_call_rate`
   - 或设置约束：`false_call_rate <= 0.025` 后优先 `chi/pon` recall。
2. 做权重对照实验。当前 `chi/pon=8`、`ron=12` 让模型能学到 response，但 epoch 10 已偏保守。建议跑：
   - `pass=1, chi/pon=10, ron=12`
   - `pass=0.75, chi/pon=8, ron=12`
   - 保持过滤 `candidate_count >= 2` 不变。
3. 增强 response action 特征，优先表达：
   - chi pattern / consumed tile 位置。
   - 是否消耗赤牌、dora、役牌对子、关键搭子。
   - call 后向听数、有效牌变化、手牌形状变化。
   - source 玩家、巡目、点差和亲家状态。
4. eval 报告增加 tie 统计：记录 label score 与 top score 是否相等、同分 top1 数量、exact argmax top1 与 tie-aware top1 的差异。
5. 对 false call / missed call 各抽样 50 条，用 `inspect-sample` 和 `dump-case` 检查是否是特征缺失、标签风格差异，还是规则/样本构建边界。

短期推荐保存两条 response baseline：

- 保守模型：当前 epoch 10 `best.pt`，false call 低，适合先接入推理安全线。
- 积极模型：epoch 3 或后续 recall-oriented 配置，适合分析 `chi/pon` recall 上限。
