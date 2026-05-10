# Turn / Response 训练分析记录

更新时间：`2026-05-06`

本文记录 `turn/response` 从 4k local smoke、response 训练与特征增强，到 2026-05-06 P800 大数据单任务 baseline 的训练结果分析，以及当前下一步改进方向。

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
3. 训练工程先补齐真正的 8 卡 DDP、grad accumulation、step eval/save；`bf16` autocast 已在单卡训练路径接入。
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

## 8. 2026-05-03 response action 特征增强

本轮按“先补候选动作特征，不改模型结构”的方向实现了第一版 response candidate feature 扩展。

新增内容：

- `RESPONSE_ACTION_DIM` 增加一组 `RESPONSE_CALL_FEATURE_DIM`。
- response 样本构建阶段离线写入增强后的 `candidate_features_bin`。
- 新增 response 候选特征包括：
  - call 前当前向听数。
  - call 后按固定副露数计算的标准形向听数。
  - `shanten_delta`。
  - call 前后有效牌种类数、有效牌枚数及 delta。
  - 是否消耗赤牌、dora、目标牌是否 dora。
  - 是否消耗对子、是否拆搭子。
  - `pon` 是否碰役牌。
  - `chi` 形状：两面、嵌张、边张。
  - call 后面子数、对子数、两面、坎张/边张、孤张摘要。
  - 巡目和剩余巡目。

实现约束：

- 不改变合法动作生成。
- 不改变模型结构。
- 不改变 response 训练过滤策略。
- 继续使用离线预编码，训练期不重新计算规则特征。

重要兼容性：

由于 `RESPONSE_ACTION_DIM` 已变化，旧 `data/silver_samples` 中的 response `candidate_features_bin` 会和当前代码不匹配。下一轮训练前必须重建：

```bash
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 32 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
```

建议下一次实验只比较“特征增强前后”的差异，先不要同时改 class weight：

- 配置保持当前 `response_local_smoke.yaml`。
- 训练 `10` epoch。
- 重点观察：
  - `multi_candidate top1`
  - `multi_candidate macro top1`
  - `chi top1`
  - `pon top1`
  - `non_pass_f1`
  - `false_call_rate`
  - `missed_call_rate`

如果增强特征有效，期望是：

- `multi_candidate macro top1` 从 `0.7162` 往 `0.74+` 提升。
- `chi top1` 从 `0.58` 附近往 `0.62+` 提升。
- `pon top1` 从 `0.83` 附近往 `0.86+` 提升。
- `false_call_rate` 维持在 `2.5%` 以下。

## 9. 2026-05-03 response 特征增强后复训结果

本轮是第 8 节 response action feature 增强后的复训结果。训练配置保持 `configs/response_local_smoke.yaml` 不变：

- checkpoint 目录：`outputs/checkpoints/response_local_smoke`
- 训练配置：`hidden_size=128`，`batch_size=1024`，`epochs=10`
- 训练过滤：`response_min_train_candidate_count=2`
- 训练样本：从 `1490034` 条 response train 样本过滤到 `141423` 条，过滤 `1348611` 条 pass-only 单候选样本。
- 最新复评报告：
  - `outputs/checkpoints/response_local_smoke/eval_valid_best_feature_2026-05-03.json`
  - `outputs/checkpoints/response_local_smoke/eval_test_best_feature_2026-05-03.json`

注意：`eval_valid_best_2026-05-03.json` 和 `eval_test_best_2026-05-03.json` 是 19:41 生成的旧报告，早于本轮 21:55 的最新 checkpoint，不应再作为本轮结论。

### 9.1 与特征增强前对比

增强前结果指第 7 节中过滤 pass-only 后、但尚未补 response action feature 的 2026-05-03 复训结果；增强后结果来自本轮最新 `best.pt`。

| 指标 | 增强前 valid | 增强后 valid | 增强前 test | 增强后 test |
|---|---:|---:|---:|---:|
| overall top1 | `0.9788` | `0.9859` | `0.9793` | `0.9859` |
| macro top1 | `0.7515` | `0.8107` | `0.7597` | `0.8925` |
| multi-candidate top1 | `0.7766` | `0.8521` | `0.7804` | `0.8502` |
| multi-candidate macro top1 | `0.7162` | `0.7882` | `0.7248` | `0.8697` |
| non-pass precision | `0.4962` | `0.6228` | `0.5040` | `0.6210` |
| non-pass recall | `0.8034` | `0.8575` | `0.8114` | `0.8534` |
| non-pass F1 | `0.6135` | `0.7215` | `0.6218` | `0.7189` |
| false call rate | `0.0174` | `0.0111` | `0.0170` | `0.0111` |
| missed call rate | `0.1966` | `0.1425` | `0.1886` | `0.1466` |
| chi top1 | `0.5822` | `0.6455` | `0.5778` | `0.6303` |
| pon top1 | `0.8334` | `0.8963` | `0.8434` | `0.8895` |
| ron top1 | `1.0000` | `1.0000` | `1.0000` | `1.0000` |

结论：

- response action feature 增强有效，并且超过了第 8 节的预期目标：valid `multi_candidate macro top1=0.7882`、`chi top1=0.6455`、`pon top1=0.8963`，同时 `false_call_rate=1.11%`。
- valid/test 一致性很好：test `multi_candidate top1=0.8502`、`non_pass_f1=0.7189`、`false_call_rate=1.11%`，说明这轮提升不是只吃到了 valid split。
- 这轮不再是“降低 false call 但明显牺牲漏叫”的状态。增强前 valid `missed_call_rate=19.66%`，增强后降到 `14.25%`；test 也从 `18.86%` 降到 `14.66%`。
- `ron` 仍然满分，核心难点继续集中在 `chi/pon` 是否值得叫，以及 pass hard negative。

`macro top1` 尤其 test 的 `0.8925` 会受 `chankan/daiminkan` 这类极少样本波动影响，后续判断 response 质量仍应优先看 `multi_candidate`、`non_pass F1`、`false_call_rate`、`missed_call_rate`、`chi/pon top1`。

### 9.2 epoch 与 checkpoint 选择

最新 `train_metrics.json` 显示 `best.pt`、`best_response.pt`、`best_accuracy.pt` 都选择 valid epoch 10：

| 指标 | epoch 10 valid |
|---|---:|
| valid loss | `0.0954` |
| overall top1 | `0.9859` |
| macro top1 | `0.8104` |
| multi-candidate top1 | `0.8521` |
| multi-candidate macro top1 | `0.7882` |
| non-pass F1 | `0.7215` |
| false call rate | `0.0111` |
| missed call rate | `0.1425` |
| response score (`non_pass_f1 - false_call_rate`) | `0.7104` |

训练曲线的主要变化是：随着 epoch 增加，模型从“更积极叫牌”变成“更会区分 pass hard negative”。epoch 1 的 `missed_call_rate=0.0741` 更低，但 `false_call_rate=0.0240`、`non_pass_precision=0.4514`，误叫偏多；epoch 10 将 false call 压到 `0.0111`，同时 non-pass F1 提到 `0.7215`。因此当前 checkpoint selection 对本轮是合理的。

仍需保留的风险是：如果后续目标是更积极的副露风格，单纯最大化 `non_pass_f1 - false_call_rate` 可能仍偏保守。下一轮在 20 万局上先不要改 selection，等拿到大数据 baseline 后再决定是否加入 `chi/pon recall` 下限。

### 9.3 当前错误结构

最新 valid confusion：

| label -> pred | count |
|---|---:|
| `pass -> chi` | `967` |
| `pass -> pon` | `1015` |
| `pass -> ron` | `60` |
| `pass -> daiminkan` | `32` |
| `chi -> pass` | `383` |
| `pon -> pass` | `181` |
| `chi -> chi` | `810` |
| `pon -> pon` | `1582` |
| `ron -> ron` | `1019` |

最新 test confusion：

| label -> pred | count |
|---|---:|
| `pass -> chi` | `1014` |
| `pass -> pon` | `1009` |
| `pass -> ron` | `64` |
| `pass -> daiminkan` | `34` |
| `chi -> pass` | `395` |
| `pon -> pass` | `201` |
| `chi -> chi` | `781` |
| `pon -> pon` | `1642` |
| `ron -> ron` | `1030` |

相比增强前，false call 与 missed call 同时下降。剩余错例中，`pass -> chi/pon` 仍是最大量，说明下一阶段最有价值的错例分析不是再盲目加特征，而是抽样看 false call 是否符合牌谱玩家风格，还是模型缺少防守/点况/他家威胁信息。

### 9.4 是否可以开始 20 万局训练

可以开始，但建议把“开始 20 万局训练”限定为：**全量数据构建 + 当前单任务 baseline 训练**，而不是直接进入 8 卡 DDP、多任务联合训练或复杂模型。

理由：

- 数据链路已经在 4233 局上验证过 `files_failed=0`、非法标签率 `0.00%`。
- response 最新本地复训已经通过目标门槛：`multi_candidate top1≈0.85`、`non_pass_f1≈0.72`、`false_call_rate≈1.1%`，valid/test 稳定。
- response action feature 维度已变，20 万局训练本来就必须重建 `silver_samples`，现在正适合作为新的数据版本基线。
- 单卡训练路径已接入 `train.precision: bf16`、CUDA TF32 和低同步指标聚合；如果 `data_wait≈0`，P800 调优应优先看 `samples_per_s` / `avg_compute_time`，再调 `batch_size`。
- 当前工程尚未真正实现 8 卡 DDP / 多任务 batch 混合；`configs/train/p800_ddp.yaml` 仍应视为配置骨架，不应把分布式工程风险和模型效果评估混在同一轮。

建议 20 万局第一阶段命令：

```bash
python -m mjai.cli preprocess --input data/raw_xml --output data/bronze_events
python -m mjai.cli validate-events --input data/bronze_events --report outputs/reports/event_validation_20w.json
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 32 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests

python -m mjai.cli train --config configs/response_p800.yaml
python -m mjai.cli eval --ckpt runs/2026-05-06_p800_baseline/response/checkpoints/best.pt --split valid --report runs/2026-05-06_p800_baseline/response/eval/eval_valid_best.json
python -m mjai.cli eval --ckpt runs/2026-05-06_p800_baseline/response/checkpoints/best.pt --split test --report runs/2026-05-06_p800_baseline/response/eval/eval_test_best.json
```

P800 上 `build-dataset --workers` 可从 `32` 起步，再根据 CLI 输出的 `samples_per_s` 试 `48` 或 `64`。第一轮 response baseline 先用 `configs/response_p800.yaml`，不要同时改 class weight、selection score 或模型结构。

20 万局 response baseline 的最低验收建议：

- `validate-events` 的 `files_failed=0` 或失败原因可解释且占比极低。
- `label_action not in legal_actions` 继续为 `0`。
- valid/test `multi_candidate top1` 不低于本地基线太多，建议先看 `>=0.84`。
- valid/test `non_pass_f1 >= 0.70`。
- `false_call_rate <= 0.015`。
- `missed_call_rate <= 0.16`。
- `chi top1 >= 0.62`，`pon top1 >= 0.88`。

如果 20 万局的分布和当前 4233 局不同，阈值可以放宽，但必须解释差异来自数据分布还是模型退化。

### 9.5 下一阶段任务

优先级建议如下：

1. 先跑 20 万局数据构建，产出 `preprocess_summary.json`、`event_validation_20w.json`、`build_summary.json` 和 manifest split 统计。
2. 用当前 `response_p800.yaml` 跑 response 单任务 baseline，并固定保存 valid/test eval report。
3. 再跑 `turn_p800.yaml`，建立同一份 20 万局数据上的 turn baseline。
4. 对 response false call / missed call 各抽样 `50-100` 条，用 `inspect-sample` 与 `dump-case` 聚合错误来源，重点看防守信息、点况信息和玩家风格差异。
5. 在 20 万局单任务 baseline 稳定后，再实现真正的 `task: multitask`、DDP、grad accumulation 和 step eval/save。

短期不要继续在 4k 局上反复调 response 权重。当前 4k 结果已经足够支持进入 20 万局 baseline；继续小数据调参会开始过度贴合当前 split。

## 10. 2026-05-06 P800 大数据 baseline 结果

本节记录本轮 P800 训练产物。原始训练输出曾落在 `outputs/`，现已按 run 归档：

- `runs/2026-05-06_p800_baseline/run_summary.json`
- `runs/2026-05-06_p800_baseline/turn/checkpoints`
- `runs/2026-05-06_p800_baseline/response/checkpoints`
- `runs/2026-05-06_p800_baseline/turn/logs`
- `runs/2026-05-06_p800_baseline/response/logs`

本轮数据版本摘要已归档到可复用数据集目录 `runs/_datasets/tenhou_20w_2026-05-06/`：

- `runs/_datasets/tenhou_20w_2026-05-06/dataset_summary.json`
- `runs/_datasets/tenhou_20w_2026-05-06/preprocess_summary.json`
- `runs/_datasets/tenhou_20w_2026-05-06/event_validation_20w.json`
- `runs/_datasets/tenhou_20w_2026-05-06/build_summary.json`
- `runs/_datasets/tenhou_20w_2026-05-06/manifest_summary.json`
- `runs/_datasets/tenhou_20w_2026-05-06/train.jsonl`
- `runs/_datasets/tenhou_20w_2026-05-06/valid.jsonl`
- `runs/_datasets/tenhou_20w_2026-05-06/test.jsonl`

下面的数据规模、校验结果、split 和动作分布来自该 run 目录，并已和 P800 训练日志、`train_metrics.json`、valid/test eval report 交叉核对。

### 10.1 数据版本与 split 核对

`preprocess_summary.json`：

- games：`212163`
- events：`215224122`
- 平均每个 game 事件数：约 `1014.4`
- warnings：空

`event_validation_20w.json`：

- `files_seen=212163`
- `files_valid=212163`
- `files_failed=0`
- `total_events=215224122`
- `total_kyokus=2142692`
- `total_discards=102949870`
- `total_melds=4313044`
- `error_counts={}`
- warnings：空

`build_summary.json`：

- turn 样本：`103874653`
- response 样本：`309055971`
- 构建耗时：约 `2.72` 小时
- 构建吞吐：`42109.5 samples/s`，`21.64 games/s`
- `games_with_zero_turn_samples=0`
- `games_with_zero_response_samples=0`
- warning 只有 `replay validation skipped during dataset build`，符合先单独 validate、构建时 `--skip-validation` 的流程

`manifest_summary.json`：

| split | games | turn samples | response samples |
|---|---:|---:|---:|
| train | `169728` | `83077288` | `247177362` |
| valid | `21225` | `10404270` | `30955527` |
| test | `21210` | `10393095` | `30923082` |

三份 manifest 逐行累加后的 `turn_rows` / `response_rows` 与 `manifest_summary.json` 完全一致，且 `build_summary` 总样本数等于 train/valid/test 三个 split 之和：

- turn：`103874653 = 83077288 + 10404270 + 10393095`
- response：`309055971 = 247177362 + 30955527 + 30923082`
- games：`212163 = 169728 + 21225 + 21210`

这也和 `turn_p800` / `response_p800` 训练日志、`train_metrics.json` 记录的样本数完全对上。因此本轮 checkpoint 已经能追溯到明确的数据版本。

### 10.2 动作分布

本轮大数据动作分布：

| turn label | count | pct |
|---|---:|---:|
| discard | `101380656` | `97.5990%` |
| riichi_discard | `1569214` | `1.5107%` |
| tsumo | `734347` | `0.7070%` |
| ankan | `98535` | `0.0949%` |
| kakan | `68787` | `0.0662%` |
| kyuushu_kyuuhai | `23114` | `0.0223%` |

| response label | count | pct |
|---|---:|---:|
| pass | `303845321` | `98.3140%` |
| pon | `2394152` | `0.7747%` |
| chi | `1738957` | `0.5627%` |
| ron | `1064608` | `0.3445%` |
| daiminkan | `12613` | `0.0041%` |
| chankan | `320` | `0.0001%` |

和 4k local 数据相比，turn 分布整体接近，但 response 的 `pass` 占比从 `97.9326%` 升到 `98.3140%`，`ron/pon/chi` 占比都有所下降。这解释了为什么本轮 response 的 recall 变高但 precision 仍有压力：真实可叫样本更稀，false call 的绝对数会被海量 pass 放大。

另一个需要记住的差异是样本密度：旧 `4233` 数据每 game 约 `149.1` 条 turn、`443.4` 条 response；本轮每 game 约 `489.6` 条 turn、`1456.7` 条 response。新数据更接近完整对局尺度，旧 4k local 不适合作为每局长度分布的代表，只适合作为小规模功能和相对指标基线。

### 10.3 本轮训练规模与配置

| 任务 | checkpoint | 训练配置要点 | 有效 train 样本 | valid 样本 | test 样本 |
|---|---|---|---:|---:|---:|
| turn | `runs/2026-05-06_p800_baseline/turn/checkpoints/best.pt` | `hidden_size=512`，`batch_size=8192`，`epochs=10`，`candidate_limit=64`，`bf16` | `83077288` | `10404270` | `10393095` |
| response | `runs/2026-05-06_p800_baseline/response/checkpoints/best.pt` | `hidden_size=256`，`batch_size=8192`，`epochs=10`，`candidate_limit=16`，`bf16`，训练过滤 `candidate_count >= 2` | `23120538` | `30955527` | `30923082` |

response 的原始 train 样本为 `247177362`，训练过滤掉 `224056824` 条 pass-only 单候选样本，实际保留 `23120538` 条，保留率约 `9.35%`。这是符合预期的：valid/test 仍保留全量样本，其中单候选 pass-only 约占 `90.64%`，所以 response 的 overall accuracy 仍然不能单独作为质量判断。

P800 单卡吞吐基本正常：

- turn 训练窗口大多在 `49k-55k samples/s`，`data_wait` 通常远低于 `compute_time`。
- response 训练窗口大多在 `65k samples/s` 左右，除每个 epoch 首 batch 初始化外，DataLoader 也不是主要瓶颈。

因此下一步性能调优不应优先继续加 DataLoader worker；如果要提速，优先试更大的 batch、mmap/更少小文件开销，或提高模型计算密度。

### 10.4 Turn 结果

`turn_p800/best.pt` 的离线 eval 结果：

| 指标 | 4k local valid | P800 valid | P800 test |
|---|---:|---:|---:|
| loss | `1.0986` | `0.8204` | `0.8206` |
| top1 | `0.6318` | `0.7007` | `0.7008` |
| top3 | `0.9088` | `0.9444` | `0.9446` |
| top5 | `0.9682` | `0.9853` | `0.9853` |
| macro top1 | `0.7662` | `0.9134` | `0.9134` |
| discard top1 | `0.6266` | `0.6957` | `0.6958` |
| riichi_discard top1 | `0.7819` | `0.8542` | `0.8534` |
| ankan top1 | `0.3889` | `0.9629` | `0.9602` |
| kakan top1 | `0.8000` | `0.9689` | `0.9726` |
| tsumo top1 | `1.0000` | `0.9986` | `0.9983` |

结论：

- 大数据 baseline 对 turn 明显有效。相对 4k local，exact top1 提升约 `6.9` 个百分点，top3 提升约 `3.6` 个百分点，macro top1 提升约 `14.7` 个百分点。
- valid/test 几乎重合，说明这轮没有明显只吃 valid split；相比 4k local 的过拟合状态，大数据已经让训练更稳定。
- 少数类明显稳定，尤其 `ankan/kakan/riichi_discard`。这说明之前少数类波动主要是数据量不足，不是规则或 head 完全不可学。
- 主要瓶颈仍是 discard exact tile。valid `discard top1=0.6957`，但 `discard top3=0.9432`、`top5=0.9849`，说明模型多数时候能把合理弃牌排进前几名，缺的是把 top2/top3 推到 top1 的上下文信息。

候选规模也支持这个判断。valid 中最大量的是 `9-16` 候选桶，样本 `9145517`，top1 只有 `0.6884`，但 top3 已有 `0.9408`。这类样本基本就是“多个可弃牌候选很接近”的决策面，继续只调 loss 或训练轮数，收益会有限。

下一轮 turn 的重点不应先上复杂结构，而应围绕 discard 决策补上下文特征和错例聚合：

1. 抽样 label rank 为 `2-3` 的 discard 错例，按局面阶段、候选牌种、是否 dora/赤、是否安全牌、是否影响向听/有效牌聚合。
2. 补牌河顺序、摸切/手出、各家现物/筋/one-chance、他家立直/副露威胁、点况和剩余巡目等特征。
3. 对 `best_loss.pt`、`best_accuracy.pt`、`best_macro.pt` 固定导出 valid/test report，确认当前 `best.pt` 选择是否也是 turn 的最优实战折中。

### 10.5 Response 结果

`response_p800/best.pt` 的离线 eval 结果：

| 指标 | 4k feature test | P800 valid | P800 test |
|---|---:|---:|---:|
| overall top1 | `0.9859` | `0.9871` | `0.9871` |
| macro top1 | `0.8925` | `0.8944` | `0.8974` |
| multi-candidate top1 | `0.8502` | `0.8623` | `0.8619` |
| multi-candidate macro top1 | `0.8697` | `0.8712` | `0.8742` |
| non-pass precision | `0.6210` | `0.5769` | `0.5761` |
| non-pass recall | `0.8534` | `0.9349` | `0.9356` |
| non-pass F1 | `0.7189` | `0.7135` | `0.7131` |
| false call rate | `0.0111` | `0.0118` | `0.0118` |
| missed call rate | `0.1466` | `0.0651` | `0.0644` |
| chi top1 | `0.6303` | `0.8352` | `0.8376` |
| pon top1 | `0.8895` | `0.9531` | `0.9528` |
| ron top1 | `1.0000` | `0.9985` | `0.9984` |
| daiminkan top1 | 未重点看 | `0.5914` | `0.6076` |

结论：

- 本轮 response 已通过之前设定的 20 万局 baseline 验收线：valid/test `multi_candidate top1 >= 0.84`、`non_pass_f1 >= 0.70`、`false_call_rate <= 0.015`、`missed_call_rate <= 0.16`、`chi top1 >= 0.62`、`pon top1 >= 0.88`。
- 大数据最明显的收益是 `chi/pon` recall 和 exact action：test `chi top1` 从 `0.6303` 提到 `0.8376`，`pon top1` 从 `0.8895` 提到 `0.9528`，missed call rate 从 `14.66%` 降到 `6.44%`。
- 代价是 non-pass precision 下降：test 从 `0.6210` 降到 `0.5761`。也就是说模型更愿意叫牌，漏叫显著减少，但误叫压力没有消失。
- false call rate 仍然只有约 `1.18%`，看起来不高；但因为 pass 样本极多，test 绝对量是 `359516` 条 false call，其中主要是 `pass->chi=181825`、`pass->pon=168229`。如果直接接入推理，这会变成明显偏激进的副露风格。
- `ron` 仍接近满分，说明规则层提供的和牌机会信号已经足够强。`daiminkan` 仍弱，但样本很少，短期不应让它支配主线决策。

checkpoint 选择方面，`best.pt` / `best_response.pt` 仍按 `non_pass_f1 - false_call_rate` 选择。本轮 valid epoch 8 的 response score 最高，虽然 epoch 10 的 loss 更低、missed call 更低，但 false call 更高、F1 略低。因此当前 best 选择是合理的；如果后续想要更保守或更激进，应显式做阈值/偏置扫描，而不是隐式换 epoch。

### 10.6 当前总体判断

这轮 P800 大数据 baseline 是成功的：

- turn：解决了 4k local 下明显过拟合和少数类不稳定的问题，建立了 `top1≈0.70 / top3≈0.944 / macro≈0.913` 的稳定基线。
- response：在 pass-only 过滤和 action feature 增强基础上，进一步显著改善 `chi/pon`，并保持 valid/test 一致。
- 数据版本：`runs/2026-05-06_p800_baseline/` 中的 preprocess/validation/build/manifest 摘要与 checkpoint 训练样本数完全一致。
- 训练工程：单卡 `bf16` 路径可用，吞吐稳定，日志足够判断 DataLoader 不是主要瓶颈。

但它还不是可以直接进入复杂模型或多任务训练的信号。现在最有价值的问题已经从“能不能学到”变成：

- turn 如何把 discard top3 内的候选重新排序到 top1。
- response 如何在保持高 recall 的同时控制 pass hard negative 上的 false call。
- 后续 run 管理如何把 dataset summary、manifest、config、checkpoint 索引和 eval report 固定到可追溯目录结构里，避免再次靠人工对齐。

### 10.7 下一步建议

优先级从高到低：

1. 后续继续使用 `runs/<date>_<exp>/` 结构，并保留轻量 `run_summary.json`，记录 `dataset_id`、checkpoint 路径、配置路径和 eval report 路径；数据集级摘要和 manifest 统一放在 `runs/_datasets/<dataset_id>/`，大文件本身继续不进 git。
2. 对 response 做 pass/call 校准实验。固定当前 `response_p800/best.pt`，在 valid 上扫描 `pass` logit bias 或 score margin，输出不同 false call 目标下的 `non_pass precision/recall/F1`、`missed_call_rate`、`chi/pon top1`。建议至少看 `false_call_rate <= 0.010 / 0.008 / 0.005` 三档；当前仓库已提供 `calibrate-response` 命令，下一步应直接在 P800 baseline checkpoint 上跑 valid/test 校准报告。
3. 对 response 错例做抽样。使用 `sample-response-errors` 分别抽 `pass->chi`、`pass->pon`、`chi->pass`、`pon->pass` 各 `100` 条，用 `inspect-sample` 和 `dump-case` 判断误差来自玩家风格、点况/防守信息缺失，还是 action feature 仍不足。
4. 对 turn 做 top-k 错例聚合。使用 `sample-turn-errors` 重点抽 label rank 为 `2/3` 的 discard 样本，而不是泛泛抽所有错误；这些样本最能说明缺哪些排序特征。
5. 再补 turn 特征，不先换复杂模型。第一步牌河时序 V1 已落地到 discard 候选特征，`TURN_ACTION_DIM=85`；下一步重建数据并固定当前训练配置复跑，观察 `discard top1` 是否提升而 `top3` 基本稳定。后续再评估摸切/手出、对手威胁、候选弃牌安全性、dora/赤损失、点况和局面阶段。
6. 对 P800 配置做一次小范围吞吐试验。response 当前运行产物使用 `batch_size=8192`，仓库配置已倾向更大 batch；可以试 `16384/32768`，只以 `samples_per_s`、显存和 eval 指标为依据。
7. 多任务、DDP 和复杂模型继续后置。等 turn 特征和 response 校准有明确结论后，再实现 `task: multitask`、step eval/save、grad accumulation 和 DDP，避免把工程变量和模型质量变量混在同一轮。

### 10.8 response 校准执行规范

本轮新增的 `calibrate-response` 不重训模型，只改变推理时的 pass/call operating point：每个响应窗口先取最高分 `pass` 和最高分 non-pass，只有当

```text
best_non_pass_score > pass_score + pass_bias
```

才选择叫牌。`pass_bias` 越大，模型越保守，false call 通常下降，但 missed call 会升高。因此它适合回答“当前 checkpoint 在不同误叫目标下还能保留多少叫牌召回”，并为后续混合训练或推理默认阈值提供约束。

P800 baseline 的第一步命令：

```powershell
python -m mjai.cli calibrate-response --ckpt runs/2026-05-06_p800_baseline/response/checkpoints/best.pt --split valid --manifest runs/_datasets/tenhou_20w_2026-05-06/valid.jsonl --report runs/2026-05-06_p800_baseline/response/calibration/response_calibration_valid_best.json --pass-biases 0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.30,0.40,0.60,0.80,1.00 --false-call-targets 1%,0.8%,0.5%
```

报告会写出：

- `points`：每个 `pass_bias` 下的 `false_call_rate`、`missed_call_rate`、`non_pass_precision/recall/F1`、`multi_candidate_top1_accuracy` 和按 label kind 的 top1。
- `selections`：对每个 false call 目标，先过滤 `false_call_rate <= target`，再选择 `non_pass_f1` 最高的 bias。

valid 上选出 bias 后，再在 test 上复核一次，不用重新训练：

```powershell
python -m mjai.cli calibrate-response --ckpt runs/2026-05-06_p800_baseline/response/checkpoints/best.pt --split test --manifest runs/_datasets/tenhou_20w_2026-05-06/test.jsonl --report runs/2026-05-06_p800_baseline/response/calibration/response_calibration_test_best.json --pass-biases <valid_selected_biases> --false-call-targets 1%,0.8%,0.5%
```

这一步产物仍属于本地 run 证据，`runs/` 目录继续由 `.gitignore` 忽略；文档只记录结论、命令和关键指标。

### 10.9 response 校准 valid 结果

`runs/2026-05-06_p800_baseline/response/calibration/response_calibration_valid_best.json` 已生成。报告覆盖 P800 valid 全量 `30,955,527` 条 response 样本，其中 `actual_pass=30,432,678`、`actual_non_pass=522,849`、`multi_candidate_samples=2,898,544`。

关键扫描点如下：

| pass_bias | false call | missed call | non-pass P/R/F1 | multi top1 | chi top1 | pon top1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `1.18%` | `6.51%` | `57.7 / 93.5 / 71.4` | `86.2%` | `83.4%` | `95.3%` |
| `0.2` | `1.06%` | `7.65%` | `59.9 / 92.3 / 72.7` | `87.3%` | `81.4%` | `94.4%` |
| `0.4` | `0.95%` | `8.92%` | `62.2 / 91.1 / 73.9` | `88.2%` | `79.2%` | `93.4%` |
| `0.6` | `0.85%` | `10.34%` | `64.5 / 89.7 / 75.0` | `89.0%` | `76.9%` | `92.1%` |
| `0.8` | `0.75%` | `11.89%` | `66.9 / 88.1 / 76.0` | `89.8%` | `74.5%` | `90.7%` |
| `1.0` | `0.66%` | `13.68%` | `69.2 / 86.3 / 76.8` | `90.4%` | `71.7%` | `89.0%` |

结论：

- `pass_bias=0.4` 是第一个进入 `false_call_rate <= 1.0%` 的点：false call 从 `358,501` 降到 `289,237`，但 missed call 从 `34,018` 增到 `46,613`；`chi/pon top1` 仍保留在 `79.2% / 93.4%`。这是当前最合理的“平衡默认值”候选。
- `pass_bias=0.8` 是第一个进入 `false_call_rate <= 0.8%` 的点：false call 降到 `228,212`，missed call 增到 `62,177`，`chi/pon top1` 降到 `74.5% / 90.7%`。这是偏保守但仍可接受的候选。
- 报告的 `selections` 对 `1.0%` 和 `0.8%` 都选了 `pass_bias=1.0`，因为当前选择策略只在满足 false call 目标后最大化二分类 `non_pass_f1`。这个点 false call 最低、F1 最高，但相对 `pass_bias=0.0` 会让 `chi top1` 下降约 `11.7` 个百分点、`pon top1` 下降约 `6.3` 个百分点；不应直接当作唯一默认值。
- 当前扫描到 `pass_bias=1.0` 仍未达到 `false_call_rate <= 0.5%`。如果必须评估 `0.5%` 档，需要追加 `1.2 / 1.4 / 1.6 / 2.0` 等更高 bias 扫描，但预期会进一步增加 missed call 并压低 `chi/pon` exact top1。

test split 复核命令：

```powershell
python -m mjai.cli calibrate-response --ckpt runs/2026-05-06_p800_baseline/response/checkpoints/best.pt --split test --manifest runs/_datasets/tenhou_20w_2026-05-06/test.jsonl --report runs/2026-05-06_p800_baseline/response/calibration/response_calibration_test_selected.json --pass-biases 0,0.4,0.8,1.0 --false-call-targets 1%,0.8%,0.5%
```

### 10.10 response 校准 test 复核结果

`runs/2026-05-06_p800_baseline/response/calibration/response_calibration_test_selected.json` 已生成。报告覆盖 P800 test 全量 `30,923,082` 条 response 样本，其中 `actual_pass=30,400,805`、`actual_non_pass=522,277`、`multi_candidate_samples=2,895,681`。

test 四个复核点如下：

| pass_bias | false call | missed call | non-pass P/R/F1 | multi top1 | chi top1 | pon top1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `1.183%` | `6.44%` | `57.6 / 93.6 / 71.3` | `86.2%` | `83.6%` | `95.3%` |
| `0.4` | `0.952%` | `8.83%` | `62.2 / 91.2 / 74.0` | `88.2%` | `79.5%` | `93.3%` |
| `0.8` | `0.751%` | `11.84%` | `66.9 / 88.2 / 76.0` | `89.8%` | `74.5%` | `90.7%` |
| `1.0` | `0.660%` | `13.60%` | `69.2 / 86.4 / 76.9` | `90.4%` | `71.8%` | `89.0%` |

valid/test 对比非常稳定：

| pass_bias | valid false call | test false call | valid missed | test missed | valid chi/pon | test chi/pon |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.0` | `1.178%` | `1.183%` | `6.51%` | `6.44%` | `83.4 / 95.3` | `83.6 / 95.3` |
| `0.4` | `0.950%` | `0.952%` | `8.92%` | `8.83%` | `79.2 / 93.4` | `79.5 / 93.3` |
| `0.8` | `0.750%` | `0.751%` | `11.89%` | `11.84%` | `74.5 / 90.7` | `74.5 / 90.7` |
| `1.0` | `0.659%` | `0.660%` | `13.68%` | `13.60%` | `71.7 / 89.0` | `71.8 / 89.0` |

最终建议：

- 默认 operating point 建议先用 `pass_bias=0.4`。它在 valid/test 都稳定压到 `false_call≈0.95%`，比无 bias 少约 `7` 万条 false call，同时 `chi/pon top1` 仍在 `79% / 93%` 量级。
- 如果目标是更保守的副露风格，可以使用 `pass_bias=0.8`。代价是 missed call 接近 `12%`，`chi top1` 降到 `74.5%`。
- `pass_bias=1.0` 不建议作为默认值。虽然二分类 `non_pass_f1` 最高、false call 最低，但对 `chi/pon` exact action 的损失已经明显。
- 当前没有必要为了 `0.5%` false call 继续把默认点推得更保守；只有在推理实战确认仍明显误叫时，再单独追加高 bias 扫描。

下一步主线应转为 response 错例抽样：优先抽 `pass->chi`、`pass->pon`、`chi->pass`、`pon->pass` 各 `100` 条，判断误差是玩家风格差异、防守/点况信息缺失，还是 action feature 仍不够。不要从 `eval.misclassified_examples` 里抽；那只是默认最多 `10` 条的调试定位样本。

建议先按当前默认 operating point 抽 `pass_bias=0.4`：

```powershell
python -m mjai.cli sample-response-errors --ckpt runs/2026-05-06_p800_baseline/response/checkpoints/best.pt --split valid --manifest runs/_datasets/tenhou_20w_2026-05-06/valid.jsonl --output runs/2026-05-06_p800_baseline/response/error_samples/response_error_samples_valid_pb0.4.jsonl --targets "pass->chi,pass->pon,chi->pass,pon->pass" --per-target 100 --pass-bias 0.4
```

如果需要和原始 eval confusion 对齐，再补一份 raw argmax：

```powershell
python -m mjai.cli sample-response-errors --ckpt runs/2026-05-06_p800_baseline/response/checkpoints/best.pt --split valid --manifest runs/_datasets/tenhou_20w_2026-05-06/valid.jsonl --output runs/2026-05-06_p800_baseline/response/error_samples/response_error_samples_valid_raw.jsonl --targets "pass->chi,pass->pon,chi->pass,pon->pass" --per-target 100 --pass-bias 0
```

`--seed` 默认继承 checkpoint config 的全局 `seed`，需要另一组随机样本时才显式覆盖。PowerShell 会把未引用的 `>` 当成重定向符号，所以 `--targets` 必须加引号。

### 10.11 turn discard rank 2/3 错例抽样

turn 侧不要从 `eval.misclassified_examples` 里随手挑样本；那个字段仍只是默认最多 `10` 条的调试定位样本。要分析“模型差一点排对”的出牌错误，直接用 `sample-turn-errors` 重新跑 manifest 推理，并按 label kind 与 label rank 分桶抽样。

P800 baseline 建议先抽 valid：

```powershell
python -m mjai.cli sample-turn-errors --ckpt runs/2026-05-06_p800_baseline/turn/checkpoints/best.pt --split valid --manifest runs/_datasets/tenhou_20w_2026-05-06/valid.jsonl --output runs/2026-05-06_p800_baseline/turn/error_samples/turn_discard_rank23_valid.jsonl --label-kinds discard --ranks 2,3 --per-rank 100
```

如果 valid 分析已经形成明确结论，再补 test 复核：

```powershell
python -m mjai.cli sample-turn-errors --ckpt runs/2026-05-06_p800_baseline/turn/checkpoints/best.pt --split test --manifest runs/_datasets/tenhou_20w_2026-05-06/test.jsonl --output runs/2026-05-06_p800_baseline/turn/error_samples/turn_discard_rank23_test.jsonl --label-kinds discard --ranks 2,3 --per-rank 100
```

命令默认 `--label-kinds discard --ranks 2,3`，保留显式参数是为了让 run 记录自解释。输出 JSONL 每行包含 `sample_id`、`target`（例如 `discard@rank2`）、`label_rank`、label/pred action、top predictions、legal actions，以及可直接执行的 `inspect-sample` 和 `dump-case` 命令。后续人工分析应优先按候选弃牌、向听变化、有效牌变化、dora/赤损失、危险度、局面阶段和点况聚合。

### 10.12 turn 牌河时序特征 V1

第一版 turn 特征回补已先落在 discard candidate 侧，不改样本 JSON schema，也不改 Tenhou 解析；直接利用现有 `state["all_discards"]` 中每家牌河的顺序。`TURN_ACTION_DIM` 从 `69` 增至 `85`，旧 `silver_samples` 中的 turn `candidate_features_bin` 不再匹配当前代码。

新增的 16 个特征覆盖：

- 候选弃牌在自家牌河与其他三家牌河中的出现次数。
- 有几个对手打过这张牌。
- 有几个对手最近 6 巡、前 6 巡打过这张牌。
- 这张牌最早/最近被对手打出的对手牌河巡目。
- 是否三家现物、至少两家现物、至少一家最近 6 巡现物。
- 对下家、对面、上家分别是否现物，以及是否最近 6 巡现物。

重建与复训命令仍是：

```powershell
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 32 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
python -m mjai.cli train --config configs/turn_p800.yaml
```

验收指标优先看 `discard top1`，同时确认 `discard top3` 基本稳定；复训后再用 `sample-turn-errors` 抽 `discard@rank2/rank3` 对比错例是否少了安全性/近期现物相关混淆。

## 11. 2026-05-10 P800 improved 结果

本轮 run 归档在 `runs/2026-05-10_p800_improved/`，继续使用同一份数据版本 `tenhou_20w_2026-05-06`。本节只保留最终对比结论；逐点 eval 和 calibration 原始报告见 run 目录。

### 11.1 turn 结果

turn improved 在 valid/test 上稳定优于 2026-05-06 baseline。test 关键指标如下：

| 指标 | baseline | improved | 变化 |
| --- | ---: | ---: | ---: |
| loss | `0.8206` | `0.8000` | `-0.0206` |
| top1 | `70.08%` | `70.75%` | `+0.67pp` |
| top3 | `94.46%` | `94.93%` | `+0.48pp` |
| top5 | `98.53%` | `98.68%` | `+0.15pp` |
| discard top1 | `69.58%` | `70.24%` | `+0.66pp` |
| discard top3 | `94.34%` | `94.83%` | `+0.49pp` |
| riichi_discard top1 | `85.34%` | `87.13%` | `+1.78pp` |
| non_discard top1 | `90.49%` | `91.37%` | `+0.88pp` |

这说明牌河时序 V1 对主目标有效：discard top1/top3 同时提升，且 valid/test 变化一致。当前可以采用 improved turn checkpoint 作为新基线。

需要记录的副作用是 `ankan top1` 从 `96.02%` 降到 `89.88%`，`kakan top1` 从 `97.26%` 降到 `96.27%`。`ankan` test 只有 `9,940` 条，短期不应压过 discard 主线，但下一步应抽 `ankan` 与 `kakan` 错例，判断是否是新特征让杠类候选排序被 discard 安全性信号干扰。

### 11.2 response 结果

response improved 的总体排序、macro 指标和 pass/call 二分类 F1 有提升，但模型明显更保守。test 关键指标如下：

| 指标 | baseline | improved | 变化 |
| --- | ---: | ---: | ---: |
| overall top1 | `98.707%` | `98.770%` | `+0.062pp` |
| macro top1 | `89.74%` | `90.60%` | `+0.86pp` |
| multi-candidate top1 | `86.19%` | `86.86%` | `+0.67pp` |
| multi-candidate macro top1 | `87.42%` | `88.43%` | `+1.01pp` |
| non-pass precision | `57.61%` | `59.04%` | `+1.43pp` |
| non-pass recall | `93.56%` | `92.72%` | `-0.84pp` |
| non-pass F1 | `71.31%` | `72.14%` | `+0.83pp` |
| false call rate | `1.183%` | `1.105%` | `-0.078pp` |
| missed call rate | `6.44%` | `7.28%` | `+0.84pp` |

按条数看，test 上 false call 从 `359,516` 降到 `335,932`，少 `23,584` 条；missed call 从 `33,634` 增到 `38,033`，多 `4,399` 条。这个取舍整体提高了二分类 F1，但不是无代价提升。

动作类别上，主要回退在吃碰 exact action：

| 动作 | baseline top1 | improved top1 | 变化 |
| --- | ---: | ---: | ---: |
| chi | `83.76%` | `81.98%` | `-1.78pp` |
| pon | `95.28%` | `94.75%` | `-0.54pp` |
| daiminkan | `60.76%` | `68.17%` | `+7.41pp` |
| ron | `99.84%` | `99.83%` | 基本持平 |

因此 response improved 可以作为更保守的新候选，但不能简单理解为全面替代 baseline。是否采用它，取决于推理侧更重视降低误叫，还是更重视吃碰 exact action。

### 11.3 response 校准结论

improved 在同一 `pass_bias` 下比 baseline 误叫更少，但 missed call 更高、`chi/pon` top1 更低。test 关键点如下：

| pass_bias | model | false call | missed call | non-pass F1 | multi top1 | chi / pon top1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `0.0` | baseline | `1.183%` | `6.44%` | `71.31%` | `86.19%` | `83.6 / 95.3` |
| `0.0` | improved | `1.105%` | `7.28%` | `72.14%` | `86.85%` | `81.9 / 94.7` |
| `0.4` | baseline | `0.952%` | `8.83%` | `73.95%` | `88.19%` | `79.5 / 93.3` |
| `0.4` | improved | `0.869%` | `10.06%` | `74.80%` | `88.85%` | `77.1 / 92.4` |
| `0.8` | baseline | `0.751%` | `11.84%` | `76.04%` | `89.78%` | `74.5 / 90.7` |
| `0.8` | improved | `0.665%` | `13.62%` | `76.74%` | `90.36%` | `71.3 / 89.2` |

baseline 的默认候选 `pass_bias=0.4` 不应直接迁移到 improved；它会让 improved 过度保守。补充的 test 中段扫描 `response_calibration_test_selected.json` 结果如下：

| pass_bias | false call | missed call | non-pass F1 | multi top1 | chi / pon top1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.12` | `1.031%` | `8.04%` | `72.99%` | `87.50%` | `80.6 / 94.1` |
| `0.16` | `1.007%` | `8.31%` | `73.27%` | `87.71%` | `80.1 / 93.9` |
| `0.3` | `0.925%` | `9.29%` | `74.18%` | `88.40%` | `78.4 / 93.1` |
| `0.4` | `0.869%` | `10.06%` | `74.80%` | `88.85%` | `77.1 / 92.4` |

因此 v1.0.0 默认 response operating point 定为 `pass_bias=0.3`：它是当前中段扫描里第一个明确进入 `false_call_rate <= 1%` 的点，比 `0.4` 保留更多 `chi/pon` exact action。`pass_bias=0.16` 可作为偏召回候选，`pass_bias=0.4` 可作为保守候选，`0.8/1.0` 不作为默认候选。

### 11.4 当前决策

- turn：采用 `runs/2026-05-10_p800_improved/turn/checkpoints/best.pt` 作为 v1.0.0 基线。
- response：采用 `runs/2026-05-10_p800_improved/response/checkpoints/best.pt` 作为 v1.0.0 基线，默认 `pass_bias=0.3`。
- baseline response 仍有对照价值，尤其用于判断 `chi/pon` exact action 的回退是否可接受。
- 下一轮不要先上多任务或 DDP。当前主线应是 response operating point 定标、response 吃碰错例抽样、turn 杠类回退排查，以及基于证据决定下一组特征。

## 12. 下一步核心工作

### 12.1 接入 response operating point

目标不是继续盲目提高 `non_pass_f1`，而是把 v1.0.0 的 response 默认点接入推理/服务配置，并让后续实验都能报告同一组 operating points。

优先级：

1. 将默认 `pass_bias=0.3` 写入后续推理/服务配置设计。
2. 同时保留 raw `0.0`、偏召回候选 `0.16`、默认 `0.3`、保守候选 `0.4` 的报告口径。
3. 后续多任务或新特征实验都必须同时报告 raw 与默认 calibrated 指标，避免只看单一 checkpoint top1。

### 12.2 做 response 定向错例抽样

当前 response 的关键矛盾已经不是“能不能叫”，而是“何时不该叫”和“叫了以后 exact action 是否对”。下一步应分别抽：

- `pass->chi` / `pass->pon`：误叫来源，重点看防守、点况、安全牌、无役风险和玩家风格。
- `chi->pass` / `pon->pass`：漏叫来源，重点看 improved 更保守后漏掉了哪些收益明确的副露。
- `chi->wrong_chi` 与 `pon->wrong_pon` 如后续命令支持，也应单独抽，用于解释 exact action 回退。

错例应按 raw 和选定默认 bias 各抽一份，避免只分析阈值后的单一视角。

### 12.3 排查 turn 杠类回退

turn improved 的主线收益明确，但 `ankan` 回退稳定出现在 valid/test。下一步应扩展或复用 `sample-turn-errors`，抽 `ankan`、`kakan` 的 rank 2/3 与错误样本，确认：

- 是否被 discard 候选的牌河安全性特征压过。
- 是否只是少数类样本波动。
- 是否需要对杠类动作加独立特征或调整 `turn_class_weights`。

### 12.4 后置多任务、DDP 和复杂模型

多任务训练、8 卡 DDP、复杂模型结构都继续后置。当前每个 checkpoint 的主要收益和代价还可以通过特征、阈值和错例分析解释；过早引入多任务或分布式工程会混淆变量。
