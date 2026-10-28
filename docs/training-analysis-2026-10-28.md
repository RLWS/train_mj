# 2026-10-28 P800 full_state_v2 训练与最终评估分析

本文统一汇总训练、独立 valid/test 评估和扩展服务参数校准，以已收到的最终报告为准。

## 最终判断

完整状态改造取得了稳定的离线收益：独立 test 普通弃牌准确率提高 1.0335 个百分点，
response 多候选准确率提高 1.1044 个百分点，non-pass F1 提高 1.9592 个百分点。
valid/test 表现接近，支持保留本轮模型；这些是牌谱拟合指标，尚无实战段位提升证据。

valid 网格内 non-pass F1 最优为 **pass_bias=1.5、no_yaku_call_penalty=0**，
对应 test F1 80.4966%。建议将其作为受控实战主要候选，同时保留 1.0/0 作为较高召回候选，
现有 2.0/0.8 作为保守对照。F1 最优不等于最少错误或最优顺位；当前配置在误响应控制上仍有优势。

下一步优先检查特殊动作错例和客户端输入完整性，并开展固定参数对局比较。
无需继续扩大粗校准网格，也没有充分依据立即从头重训。立直弃牌、大明杠与胡牌选择回退仍需处理。

## 训练完整性检查

- 10 个 epoch，epoch_metrics.jsonl 有 60 行，包含每轮两任务 train/valid 和汇总 train/valid。
- train_metrics.json 的逐任务记录与日志逐条一致；所有日志 loss 和最佳模型参数均为有限值。
- 新 best_turn.pt、best_response.pt 都来自 epoch 10，model_state_dict 逐张量完全相同。
  这是两任务最佳指标出现在同一轮的正常结果；两份文件的 summary 不同。
- checkpoint 标记 full_state_v2；两路输入投影尺寸为 512×11969 和 512×12012。
- 新旧 model/task/train 解析配置一致，包括 hidden_size=512、bf16、10 epochs、学习率 0.001、
  epoch_sequential、batch size 和类别权重。eval 配置增加 distributed=false/backend=vendor。
- 379742 个 game、385081321 个事件；回放校验 files_failed=0，只表示通过现有规则校验范围。
- turn train 148561311；response 原始 train 442001364，过滤后 41366889。
- valid：turn 18604308，response 55352691；response multi-candidate 5183428。


## 训练收敛与任务波动

turn train accuracy 从 68.34% 升到 74.52%，valid 从 60.61% 升到 73.92%；
两路 train loss 整体持续下降，不能从现有曲线认定发生严重过拟合。
不过 train/valid 的加权、pass 过滤及记录时点不同，不能直接把两者 loss 差作为泛化差距。

最后几轮仍有改进，但存在明显任务/阈值波动：

| epoch | turn valid top1 | 立直弃牌 top1 | response multi top1 | non-pass F1 | false call | missed call |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 72.12% | 88.15% | 88.58% | 74.89% | 0.93% | 8.07% |
| 6 | 72.70% | 82.25% | 86.41% | 71.96% | 1.17% | 5.66% |
| 7 | 73.33% | 87.74% | 86.06% | 71.50% | 1.21% | 5.34% |
| 8 | 73.18% | 83.49% | 88.90% | 75.45% | 0.90% | 7.87% |
| 9 | 73.90% | 83.39% | 85.84% | 71.26% | 1.24% | 4.96% |
| 10 | 73.92% | 87.39% | 89.03% | 75.70% | 0.89% | 7.78% |

当前调度是每轮先训练 turn，再训练 response，然后验证两个任务。共享主干在两个任务之间的变化
是波动的一个待验证原因；日志缺少“刚训练完 turn、尚未训练 response”的验证快照，不能确认任务干扰。
后续若做实验，可先增加这一诊断时点，再考虑低学习率延长训练或交错调度，不宜同时改变多个变量。

turn overall 增益主要来自普通弃牌。立直弃牌回退约 1.22pp，non-discard 回退约 0.75pp；
response chi/pon 提升，但 daiminkan 回退约 4.57pp。大明杠 valid 只有 2317 条，应抽样看具体局面，
不能只由 macro 指标决定是否淘汰新模型。胡牌类的小幅回退也应在错例中检查。


## 独立评估证据

本次产物：`outputs/reports/parallel_eval_20261028_103319_253798/`，已归档至
`runs/2026-10-28_p800_multitask_full_state_v2/multitask/eval/`。
审阅提交：`6596a77ba6e7fe37e1bd60be5c7530ef2a52396c`。

- 六个任务均 exit_code=0，包含 turn/response/calibration 的 valid/test；报告与日志齐全。
- launcher 为六个独立进程指定可见 GPU 0–5，并在加载模型前设置当前设备；没有把同一份数据当成六份分布式结果合并。
- 配置副本显式覆盖 full_state_v2 的 valid/test manifest，避免了 checkpoint 内旧 test 路径的问题。
- turn：valid 18,604,308，test 18,681,274；response：valid 55,352,691，test 55,580,670。
- response 多候选：valid 5,183,428，test 5,203,242。
- 历史对照使用 `2026-06-24_p800_multitask_v1_1_ddp_40w` 的 best_turn/best_response **test** 报告。
  新旧 test 样本数相同，但尚未核验逐 game_id/manifest hash，报告也未保存模型及数据 hash。
- 本次未重跑全量推理、未修改 checkpoint、未更改服务配置或原实战工作区。

这里是整轮改造的结果，包含座位归一化、分数单位修正及新状态特征等，不能把增益单独归因于某一组特征。

## test 与旧版比较

下表百分比的差值单位为百分点；false/missed 表示与牌谱 pass/non-pass 标签的差异。

| 指标 | 旧版 | full_state_v2 | 差值 |
|---|---:|---:|---:|
| turn top1 | 72.8836% | 73.8733% | +0.9897 |
| turn top3 | 96.0391% | 96.2594% | +0.2203 |
| turn top5 | 99.0790% | 99.1289% | +0.0499 |
| 普通弃牌 top1 | 72.3958% | 73.4292% | +1.0335 |
| 立直弃牌 top1 | 89.1320% | 87.8704% | -1.2616 |
| 暗杠 top1 | 90.3672% | 90.9005% | +0.5333 |
| 加杠 top1 | 96.2230% | 96.2230% | 0.0000 |
| 自摸 top1 | 99.9962% | 99.9486% | -0.0477 |
| response 多候选 top1 | 87.8678% | 88.9722% | +1.1044 |
| non-pass precision | 61.5886% | 64.0463% | +2.4577 |
| non-pass recall | 91.5609% | 92.2428% | +0.6819 |
| non-pass F1 | 73.6419% | 75.6010% | +1.9592 |
| false-call rate | 0.9867% | 0.8947% | -0.0920 |
| missed-call rate | 8.4391% | 7.7572% | -0.6819 |
| 吃精确动作 top1 | 79.4174% | 81.0314% | +1.6140 |
| 碰精确动作 top1 | 93.8641% | 94.3922% | +0.5282 |
| 大明杠 top1 | 62.3131% | 58.1354% | -4.1777 |
| 荣和 top1 | 99.9523% | 99.9310% | -0.0213 |

turn test loss 0.733000 → 0.710449；response 0.055883 → 0.052437。
在相同样本数口径下，新版少了 50,242 次 pass→non-pass、6,438 次 non-pass→pass。

新版 turn valid/test top1 为 73.8954%/73.8733%，仅差 0.0221pp；
response 多候选为 88.9987%/88.9722%，仅差 0.0265pp。
支持同分布下收益能推广到 test；这不替代不同平台、对手或实战局面的验证。

## 仍然存在的问题

### response 的总准确率容易掩盖决策误差

test 中 50,377,428 个样本只有一个候选，占 90.6384%，必然预测正确。
所以总准确率 98.9676% 不是主要实战指标。

原始模型 false-call 分母为全部 54,636,617 个 pass 标签，因此报告为 0.8947%。
若分母仅取有多个候选的 pass 标签（4,259,189），对应比例为 **11.4776%**。
这里的非 pass 包括吃碰杠和荣和，不能直接解读成真实坏鸣牌率。
488,854 次 false-call 中，pass→chi 256,490 次、pass→pon 215,445 次，占 96.54%。
吃碰阈值和局面条件是更集中的改进对象。

### 特殊动作回退要拆开看

- 立直弃牌 exact top1 同时包含“是否立直”和“立直打哪张牌”。仅有分类准确率无法确认是哪部分退步。
- 大明杠 test 只有 2,274 条，其中正确 1,322、预测 pass 865、预测 pon 85、预测 ron 2。
  回退主要表现为选择 pass，继续统一提高 pass bias 会进一步降低召回。
- 自摸标签错选从旧版 5 条升至新版 68 条；荣和标签错选从 92 条升至 133 条。
  数量不大，但价值高，应优先全量或分层复核。不能只因总体 top1 增长而忽略。
- 规则模块目前不做完整有役判定，不能直接给所有牌形完成的 tsumo/ron 候选加“必选”规则。
  同理，pass→ron 可能混合合法性/标注问题；不能都当作模型进攻失误。
- 四份 eval 各自只记录前 10 个错例，且每份都集中在单个 game，不能据此归纳全局错因。
  后续用已有 reservoir 错例采样工具覆盖更多牌局。

### turn 更像需要改善首选排序

test top1 73.87%、top3 96.26%、top5 99.13%。大量与标签不一致的局面里，标签仍排在前几名。
这支持优先抽样比较前几名在牌效、防守、打点和顺位上的差异；它不证明第 2/3 名是实战更优动作，
也不能仅凭 top3 就认定应立即上搜索或 value 网络。


## 统计口径说明

训练期 valid 与独立 eval 有小差异：训练使用 bf16 autocast，独立 eval/calibration 未启用 autocast。
精度路径不同是合理原因，但未逐样本复算，因此不把全部差异归因于这一项。

另有已确认的统计口径差异：offline `_compute_label_ranks` 只数严格高于标签的分数，
因此并列第一也算 top1 命中；calibration 则选择一个具体动作，按索引判定。
valid 的 bias=0 calibration 比 eval 少 283 个精确命中，test 少 256 个，差异均落在吃动作；
两者 pass/non-pass 混淆计数和 F1 完全一致，与同分吃候选的口径差异吻合。
最终定位具体样本仍需额外推理。这里不调整已有报告，也不把小差异误判成校准失效。
后续应同时报告 optimistic rank top1 与使用服务同分规则的 exact top1。


## 扩展校准结论

在 valid 选择的网格内 non-pass F1 最优参数为 **pass_bias=1.5、no_yaku_call_penalty=0**。
对应 test F1 为 80.4966%，比当前配置的 2.0/0.8 高 1.4287 个百分点；召回从 73.3816% 提升到 81.0004%。
这已经覆盖了实际服务配置，并解决了上次最优点落在 bias 扫描上界的问题。

建议把 1.5/0 作为下一轮独立服务对比的主要候选，1.0/0 作为保留更多吃碰召回的候选，
2.0/0.8 保留为现有保守对照。这里的最优目标是牌谱 pass/non-pass F1，尚未测得平均顺位或放铳收益，
因此不直接替换实战配置。本次只更新分析文档。

## 报告核验与证据范围

来源为 `full_state_v2_calibration_expanded_valid.json` 与同名前缀的 test.json，
已归档至 `runs/2026-10-28_p800_multitask_full_state_v2/multitask/calibration/`。
模型均为归档 `multitask/checkpoints/best_response.pt`，manifest 均为 full_state_v2。
valid/test 样本数分别为 55,352,691 / 55,580,670，与前次评估一致。

用户给出的命令为 9×4=36 点，但两份 JSON 实际均有 **9×5=45 点**，额外包含 penalty=1.0。
以下按报告实际参数分析；缺少日志不能确认额外参数来自哪个命令版本。
JSON 足以核对指标、参数网格、计数和路径，但不能确认退出码、吞吐、峰值显存及实际运行代码/模型 hash。

已核验每个组合唯一且网格完整、计数和比率自洽、关键指标有限；
与前次 14 点报告重叠的四个点（bias=0,0.3,0.6,1，penalty=0）逐字段比对一致。
valid/test 的 45 个对应点 F1 最大相差 0.1234 个百分点，主要趋势一致。
选点以 valid 为准，再查看 test 固定点，不采用 test.selections 调参。

## 关键工作点

表中单位均为百分比；false-call 指牌谱 pass 被预测成非 pass，missed-call 指反向错误。
非 pass 包含吃、碰、杠和和牌；二分类预测为非 pass 不要求选中正确的具体非 pass 动作。

| 参数 bias/penalty | split | F1 | precision | recall | false-call | missed-call |
|---|---|---:|---:|---:|---:|---:|
| 0/0 | valid | 75.6708% | 64.1433% | 92.2492% | 0.8924% | 7.7508% |
| 0/0 | test | 75.6010% | 64.0463% | 92.2428% | 0.8947% | 7.7572% |
| 0.3/0 | valid | 77.2772% | 67.3896% | 90.5652% | 0.7584% | 9.4348% |
| 0.3/0 | test | 77.2384% | 67.3260% | 90.5734% | 0.7595% | 9.4266% |
| 1/0 | valid | 79.9315% | 74.9756% | 85.5889% | 0.4944% | 14.4111% |
| 1/0 | test | 79.8976% | 74.9232% | 85.5795% | 0.4949% | 14.4205% |
| 1.2/0 | valid | 80.3256% | 77.0727% | 83.8653% | 0.4317% | 16.1347% |
| 1.2/0 | test | 80.2865% | 77.0135% | 83.8501% | 0.4324% | 16.1499% |
| 1.5/0 | valid | 80.5536% | 80.0633% | 81.0499% | 0.3493% | 18.9501% |
| 1.5/0 | test | 80.4966% | 79.9991% | 81.0004% | 0.3499% | 18.9996% |
| 2/0.8 | valid | 79.1633% | 85.7747% | 73.4982% | 0.2109% | 26.5018% |
| 2/0.8 | test | 79.0679% | 85.7094% | 73.3816% | 0.2114% | 26.6184% |

从当前 2/0.8 改为候选 1.5/0，在 test 上：

- false-call 115,506 → 191,183，增加 75,677；precision 85.7094% → 79.9991%。
- missed-call 251,292 → 179,366，减少 71,926；recall 增加 7.6188 个百分点。
- 二者简单相加的错误数反而增加 3,751，说明 F1 最优与最小二分类错误数目标不同。
- 多候选精确准确率 92.8264% → 92.7162%，略降；不能用 F1 改善代替所有指标的改善。

与 1/0 相比，1.5/0 的 test F1 只增加 0.5990pp，但 recall 从 85.5795% 降至 81.0004%。
因此 1/0 有作为实战候选的独立价值，不能仅凭 F1 一列淘汰。

三档 false-call 上限（1%、0.8%、0.5%）全部选择 1.5/0，是因为它既满足约束又有最高 F1。
三条 selection 是同一个优化结果，不代表三种策略。若目标是“在指定误响应上限下尽量保留召回”，
应改用该约束下 recall 最大的点，不能照搬 F1 选择器的结果。

## 无明显役惩罚与全局 bias 的作用

全局 bias 增加会同时压低各类非 pass 召回；无明显役惩罚只作用于被启发式标记的吃碰杠候选。
因此两个参数不能简单相加成一个全局阈值，非标记候选与荣和不会受到同样的直接惩罚。
penalty 变化也可能通过候选竞争间接改变荣和选择。

在固定 bias=2 时，test penalty 0→0.8：F1 79.8138%→79.0679%，recall 75.5751%→73.3816%，
false-call 0.2385%→0.2114%。额外惩罚确实更保守，但这一带的 F1 已走低。
在固定 bias=0 时，penalty=0.8 却能把 valid F1 从 75.6708% 提高到 77.2172%。
所以不能笼统说“无明显役惩罚无用”；它在部分工作点有效，只是本轮全网格 F1 最优不需要它。

当前 2/0.8 的 test 无明显役 false-call 仅 7,982，1.5/0 为 41,086。
因此若实战观察主要关注无明显役吃碰，二者仍有明确取舍。
该指标是启发式标记与牌谱标签的交集，不能理解为实际无役率或非法动作率。

## 具体动作召回损失

下面均为 test 的精确动作 top1，吃的多种消耗组合只有选中标签组合才算正确。

| 参数 | 吃 | 碰 | 大明杠 | 荣和 |
|---|---:|---:|---:|---:|
| 0/0 | 80.9497% | 94.3922% | 58.1354% | 99.9310% |
| 1/0 | 69.4555% | 88.8457% | 40.6772% | 99.8708% |
| 1.5/0 | 62.1818% | 84.5374% | 30.2551% | 99.8101% |
| 2/0.8 | 49.4981% | 77.6961% | 21.4600% | 99.7069% |

原始模型的大明杠已弱于旧版；统一提高 bias 会继续降低其召回。
荣和也会被全局 bias 压低。当前规则尚未完整过滤无役和牌候选，不能简单设置“见荣和必选”。
下一步应检查错例并按动作类型评估阈值；本次单一全局网格不能证明分动作阈值的收益。

## 后续执行建议

1. 当前网格已出现内部 F1 峰值，可停止扩大粗网格。若需要精细选点，未来只在 valid 的 bias 1.2–1.7、penalty 0–0.2 范围细扫；没有必要为几个小数点持续全量重扫。
2. 先对 1/0、1.5/0 与现有 2/0.8 做独立服务对比，客户端需完整提供 full_state_v2 输入。记录配置、平均顺位、四位率、放铳率、副露后和牌率和打点。
3. 校准只能改变选择取舍，不能修复立直弃牌回退、候选合法性、特征缺失或动作等价标签问题。使用下文的分层错例采样命令。
4. 本轮 test 已被多次查看；若随后继续据此选择策略，不应再把同一 test 当作最终未见数据。最终评估保留新的牌局集合或受控对局。

## 完整 45 点数据

完整 valid/test 逐点指标（含 precision、按动作类别 top1）已随归档 JSON 保存：
`runs/2026-10-28_p800_multitask_full_state_v2/multitask/calibration/full_state_v2_calibration_expanded_{valid,test}.json`。
上文"关键工作点"覆盖了选点与对照所需的全部条目；需要其余工作点时直接读取 JSON，不在本文重复展开。

## 文件指纹

以下是本地报告 SHA-256，用于识别本次分析的原始文件，不是 checkpoint hash。

- `full_state_v2_calibration_expanded_valid.json`：`7c35735c8101fd50e4270153b91faaf0df5798d75a7ace73beeeac8a2d358d23`
- `full_state_v2_calibration_expanded_test.json`：`6bcf5c48e68d69480e060a1a70bd7e26306a5142f020cd2b165c3710f93a64ff`

## 错例采样命令

在 P800 项目根目录执行，使用 valid 做诊断。turn 示例只覆盖 rank 2–5，不是全部胡牌错误。

```bash
python -m mjai.cli sample-turn-errors --ckpt runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt --split valid --config configs/eval_full_state_p800_parallel.yaml --label-kinds riichi_discard,tsumo,discard --ranks 2,3,4,5 --per-rank 100 --output outputs/reports/full_state_v2_turn_errors_valid.jsonl

python -m mjai.cli sample-response-errors --ckpt runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_response.pt --split valid --config configs/eval_full_state_p800_parallel.yaml --targets 'pass->chi,pass->pon,pass->ron,daiminkan->pass,daiminkan->pon,ron->pass,ron->chi,ron->pon' --per-target 100 --pass-bias 0 --no-yaku-call-penalty 0 --output outputs/reports/full_state_v2_response_errors_valid.jsonl
```

## 配置与验证边界

训练配置已补齐 full_state_v2 test_manifest；已有 checkpoint 内仍可能保存旧路径。
本轮并行评估通过 configs/eval_full_state_p800_parallel.yaml 正确覆盖了路径，test 结果可用于本报告。
未来独立 test 命令仍需显式指定正确配置。原配置回归测试为 6 passed。

本次重新核对原始 JSON 的 valid 选点及 test 固定点、整合文档，未重新运行全量推理或修改服务参数。
报告未提供按牌局的配对误差统计，因此不把微小参数差异解释为显著实战差异。
