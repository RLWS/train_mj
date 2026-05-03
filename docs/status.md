# 项目进度状态

更新时间：`2026-05-03`

## 1. 总体判断

当前项目已经明显超过“纯规划阶段”，可以视为：

- `Phase 0` 第一版已完成：规则与数据的最小闭环已经跑通。
- `Phase 1` 已启动：`turn/response` 单任务 baseline 都已能本地 smoke train，并可做第一版离线评估。

换句话说，仓库当前已经从“只能产样本”进入“能产样本并训练最小 baseline”，但训练体系还远没完整。

## 2. 已验证结果

以下结果已经在仓库内实际验证：

- `pytest` 全量通过，当前结果为 `95 passed`。
- `python -m mjai.cli preprocess --input tests/fixtures/tenhou --output <tmp>` 可运行。
- 对 `tests/fixtures/tenhou` 中的 `20` 个 XML 夹具实测，生成 `6281` 条事件并写入 `bronze_events`。
- `python -m mjai.cli validate-events --input <tmp> --report <tmp>` 可运行。
- 同一批事件分片实测回放结果为：
  - `files=20`
  - `valid=20`
  - `failed=0`
  - `total_events=6281`
  - 可输出事件分布、失败原因聚合、总局数/弃牌数/副露数摘要
- `python -m mjai.cli build-dataset --input <tmp> --output <tmp>` 可运行，且支持 `--workers N` 按事件分片并行构建。
- `python -m mjai.cli build-dataset --input <tmp> --output <tmp> --workers 2` 已在 Windows fixture smoke 中验证通过。
- 对 `tests/fixtures/tenhou` 中的 `20` 个 XML 夹具实测，生成：
  - `2996` 条 `turn` 样本
  - `8910` 条 `response` 样本
  - 可输出 `turn/response` 标签动作分布，其中当前真实夹具的 `turn` 已覆盖 `discard / riichi_discard / tsumo / ankan / kyuushu_kyuuhai`，构建链路本身已支持 `kakan`
- `python -m mjai.cli make-manifest --input <tmp> --output <tmp>` 可运行。
- 同一批夹具可生成 manifest，实测分桶结果为：
  - `train_games=18`
  - `valid_games=1`
  - `test_games=1`
- `python -m mjai.cli train --config <tmp>` 可运行。
- 以单局清单裁剪出的最小训练集实测，`turn` 与 `response` 任务都可完成 `1` 个 epoch 的 smoke train，并落盘：
  - `config.resolved.yaml`
  - `train_metrics.json`
  - `best.pt`
  - `best_accuracy.pt`
  - `best_loss.pt`
  - `best_macro.pt`
  - `final.pt`
- `python -m mjai.cli eval --ckpt <tmp> --split valid --report <tmp>` 可运行。
- 对最小 `turn` 与 `response` smoke checkpoint 实测，可输出：
  - `loss`
  - `top1 / top3 / top5 accuracy`
  - 按动作类别聚合的样本数与 `top-k` 指标
  - 按候选动作数量分桶的 `top-k` 指标
  - 首批错例样本的 label、top-k 预测、legal actions，以及可直接执行的 `inspect-sample` / `dump-case` 跳转命令
  - JSON 评估报告
- response 评估报告已补充 response 专用诊断：`single_candidate` / `multi_candidate`、`pass` vs `non_pass` precision / recall / F1、`false_call_rate`、`missed_call_rate`、候选集合聚合与 `label->pred` confusion。
- 2026-05-03 response 重训已完成并记录在 [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md)：训练过滤后保留 `141423 / 1490034` 条 response 训练样本；`best.pt` 在 valid/test 上分别达到 `multi_candidate top1=0.7766/0.7804`、`non_pass_f1=0.6135/0.6218`、`false_call_rate=0.0174/0.0170`。
- `python -m mjai.cli inspect-sample --dataset <tmp> --index <n>` 可运行。
- `python -m mjai.cli inspect-sample --dataset <tmp> --sample-id <id>` 可运行。
- 对生成的 `silver_samples` 实测，可按全局索引输出：
  - `sample_id / game_id / kyoku_index / step_index / actor`
  - label action
  - legal actions
  - 状态快照
  - 来源 parquet 文件与文件内行号
- `python -m mjai.cli dump-case --game-id <id> --kyoku-id <id> --step-id <id>` 可运行。
- 对生成的 `bronze_events` 实测，可按 `game_id / kyoku_index / step_index` 输出：
  - 目标事件
  - 同一局内的完整事件列表
  - 目标事件前后上下文
  - 来源 parquet 分片路径
- 真实牌谱扩大批次已在当前 `data/` 目录下完成一次端到端数据主线验证，命令链路为 `preprocess -> validate-events -> build-dataset --workers N -> make-manifest`。这批结果有记录意义：它是当前规则、样本构建与 split 逻辑在 4000+ 局真实牌谱上的第一份质量基线，可用于后续训练结果和规则回补的对照。
  - `preprocess_summary.json`：`4233` 局，`1322990` 条事件，无 warning。
  - `validate.json`：`files_seen=4233`，`files_valid=4233`，`files_failed=0`，事件校验失败率 `0.00%`；总局数 `15275`，总弃牌 `625144`，总副露 `29935`。
  - `build_summary.json`：`631016` 条 `turn` 样本，`1876854` 条 `response` 样本；`games_with_zero_turn_samples=0`，`games_with_zero_response_samples=0`，无 warning。
  - 非法标签率复核：逐行比对 `label_action_json` 是否存在于 `legal_actions_json` 后，`turn` 为 `0 / 631016 = 0.00%`，`response` 为 `0 / 1876854 = 0.00%`。
  - `turn` 标签分布：`discard=614528` (`97.3871%`)，`riichi_discard=10616` (`1.6824%`)，`tsumo=4636` (`0.7347%`)，`ankan=669` (`0.1060%`)，`kakan=474` (`0.0751%`)，`kyuushu_kyuuhai=93` (`0.0147%`)。
  - `response` 标签分布：`pass=1838052` (`97.9326%`)，`pon=17489` (`0.9318%`)，`chi=11203` (`0.5969%`)，`ron=10003` (`0.5330%`)，`daiminkan=100` (`0.0053%`)，`chankan=7` (`0.0004%`)。
  - split 规模：`train=3362` 局、`500969` 条 `turn`、`1490034` 条 `response`；`valid=426` 局、`64229` 条 `turn`、`191082` 条 `response`；`test=445` 局、`65818` 条 `turn`、`195738` 条 `response`。
  - 结论：数据构建质量可以进入训练 smoke/小规模正式训练，但类别极度不均衡已经成为训练主线的主要风险之一，整体 accuracy 不能单独作为有效模型质量判断。

## 3. 按模块看当前状态

### 已完成或基本完成

- `mjai/data/tenhou/`
  - 天凤 XML 流式解析已实现。
  - 副露解码已实现，覆盖 `chi / pon / daiminkan / ankan / kakan`；`daiminkan` 已按 Tenhou anchor tile 判定被鸣牌，避免赤牌明杠误消耗手牌。
- `mjai/rules/`
  - 手牌状态、回放状态、弃牌/吃碰杠/荣和的第一版合法动作生成已实现。
  - `turn` 合法动作已支持 `discard / riichi_discard / tsumo / ankan / kakan / kyuushu_kyuuhai`。
  - `response` 的 `ron` 已补上自家舍牌振听、同巡振听与立直后振听过滤。
  - `response` 已补上第一版 `chankan`，且 `kakan` 响应窗口不再暴露 `chi / pon / daiminkan`。
  - 基础胡牌判断与立直前听牌检查已实现。
  - 回放校验器已实现，可验证弃牌合法性和副露消耗牌一致性。
- `mjai/data/builders/`
  - `bronze_events` 预处理器已实现。
  - 事件分片回放校验器已实现。
  - `turn` 与 `response` 样本构建器已实现，当前真实夹具已产出 `tsumo`、`ankan` 与 `kyuushu_kyuuhai` 的 `turn` 样本，`kakan` 与 `chankan` 也已进入合法动作与样本构建链路。
  - `build-dataset` 已支持 `--workers N` 并行处理事件分片，优先使用进程池；当 Windows 环境触发进程池限制时会回退到线程池。
  - Parquet 写出与 manifest 生成已实现。
  - 事件分布、动作分布、split 样本统计和 warning 已实现第一版。
- `mjai/data/datasets/`
  - `build-dataset` 写出的 parquet 已包含 `state_features_bin`、`candidate_features_bin`、`label_index` 和 `label_kind`；训练和评估阶段不再逐 epoch 重新编码状态/动作特征。
  - `turn` discard 的向听数、有效牌、形状摘要等候选特征已经移到样本构建阶段离线计算，避免 DataLoader worker 在训练期反复执行 CPU 密集的规则搜索。
  - manifest 数据集会在初始化时预反序列化样本 JSON，并读取预编码特征二进制；旧 schema 的 `silver_samples` 需要重新运行 `build-dataset` / `make-manifest` 后才能训练。
- `configs/`
  - 数据、任务、模型、训练配置目录已建立。
- `mjai/features/`
  - 已有最小 `turn/response` 状态特征与动作特征编码。
- `mjai/models/`
  - 已有最小共享编码器与 `turn/response` 候选打分头。
- `mjai/training/`
  - 已有最小训练循环、`turn/response` 标签样本加权 loss、总体与宏平均 accuracy、按动作类别和主类/非主类分组的训练指标、checkpoint 落盘。
  - response 训练已支持 `train.response_min_train_candidate_count`，当前 response smoke / P800 配置默认用 `2` 过滤训练集中的 pass-only 单候选样本；valid/test 评估仍保留全量样本。
- response 的 `best.pt` / `best_response.pt` 现在按 `non_pass_f1 - false_call_rate` 选择，避免只被 overall accuracy 或 pass-only 样本支配；`best_accuracy.pt`、`best_loss.pt`、`best_macro.pt` 仍保留。
  - 2026-05-03 重训显示当前选择标准会偏向保守 response：false call 降低，但 `chi/pon` 漏叫上升。下一轮需要对 checkpoint selection、类别权重和 response action 特征做对照实验。
  - 已支持 `pin_memory`、`persistent_workers`、`prefetch_factor` 和 CUDA `non_blocking` 传输；Windows 本地 smoke 配置默认 `num_workers: 0`，避免 spawn worker 复制整份内存数据集。
  - 已支持由 `train.log_every_n_steps` 控制的 batch 进度日志，输出 batch 进度、`batch_time`、`data_wait`、`compute_time`、窗口平均耗时、samples/s、窗口 loss/accuracy 和 ETA。
- `mjai/eval/`
  - 已有第一版离线评估入口，可基于 checkpoint 复用训练配置做 split 级评估与 JSON 报告导出。
  - 已支持按动作类别、主类/非主类分组、候选规模分桶与首批错例导出。
  - response checkpoint 额外输出 `candidate_count > 1` 子集、pass/non-pass 二分类、false call / missed call 和候选集合维度指标；turn checkpoint 的 `response` 诊断字段为 `null`。
- `tests/`
  - 规则、解析、样本构建、CLI、配置和数据写出都有测试覆盖。
  - `turn/response` smoke train、offline eval、`inspect-sample` 与 `dump-case` 都有回归测试。

### 部分完成

- `mjai/cli.py`
  - 子命令框架完整。
  - `preprocess`、`validate-events`、`build-dataset`、`make-manifest`、`train`、`eval`、`inspect-sample`、`dump-case` 已可用。
  - `replay-eval`、`infer-on-state` 仍是占位实现。
- 数据链路
  - 已经落地 `raw_xml -> bronze_events -> silver_samples` 两阶段流水线。
- 第一版诊断输出已经落地，并已支持从评估错例跳转到样本与事件上下文，但还缺少更细的失败样本聚合和固定 case 管理。

### 尚未开始或基本未开始

- `mjai/serving/`

这些目录目前仍只有包骨架或说明性占位，还不能支持推理闭环。

## 4. 当前最关键的差距

### 4.1 数据流水线与基础诊断已经打通，但深入排查能力还偏弱

当前已经具备：

```text
raw_xml -> bronze_events -> validate-events -> silver_samples
```

并且已经有：

- 事件分布统计
- 错误类型聚合
- 动作标签分布
- split 样本数统计
- 小数据集空 split warning

当前已经足够进入更大规模训练试跑。后续数据诊断应围绕训练主线补强，优先输出：

- 大规模构建失败率与失败原因 Top-N
- `label_action not in legal_actions` 计数与样本定位
- 动作分布、候选规模分布和 split 规模

### 4.2 规则引擎仍是 first-pass 版本

当前规则层已经足够支撑“早期监督学习样本构建”，但还不足以称为完整日麻规则引擎。继续补规则边界的优先级应由训练数据和错例证据决定，主要候选缺口包括：

- 更完整的和牌机会放弃语义与残余振听边界
- 多家荣和等边界情况
- 更完整的立直限制与流局处理

### 4.3 训练与评估链路已打通最小版本，但覆盖面还很窄

当前已经有：

- `turn` 状态特征编码
- `response` 状态特征编码
- `turn/response` 候选动作编码
- 最小共享编码器和 `turn/response` 候选排序头
- `turn/response` 训练循环
- `loss / accuracy / macro_accuracy`
- `turn/response` 离线 `top-k`、`macro_top-k` 评估与按类别聚合报告
- checkpoint 与训练指标落盘

类别不均衡处理现状：

- 当前代码已经对 `turn` 与 `response` 都做了第一版显式处理：`mjai/training/class_balance.py` 统一读取 `train.turn_class_weights` / `train.response_class_weights`，按 batch 内 `label_kinds` 构造 `sample_weights`，再传给 `policy_loss` 做样本级加权交叉熵。
- `configs/turn_local_smoke.yaml` 已配置 `riichi_discard / tsumo / ankan / kakan / kyuushu_kyuuhai` 高于 `discard` 的权重；`configs/response_local_smoke.yaml` 已配置 `chi / pon / daiminkan / ron / chankan` 高于 `pass` 的权重，并默认过滤 response 训练集中的 `candidate_count == 1` 样本。
- 训练指标现在会在 `train_metrics.json` 与 `epoch_metrics.jsonl` 中记录 `macro_accuracy`、`per_kind` 和 `class_groups`；其中 `class_groups` 会输出 `discard/non_discard` 或 `pass/non_pass`。response 还会记录 `multi_candidate`、`non_pass_f1`、`false_call_rate`、`missed_call_rate` 等专用诊断。离线 `eval` 报告也会输出 `macro_top1/top3/top5`、按动作类别 loss/top-k、主类/非主类分组和候选规模分桶。
- 仍未解决的部分：权重仍是手工常量，不会根据 manifest 或 `build_summary.json` 自动估计；训练 DataLoader 仍是普通 shuffle，没有 `WeightedRandomSampler` 或分层采样；下一轮训练需要根据 `non_discard`、`non_pass`、`ron / chi / pon / riichi_discard / tsumo` 等指标决定是否调权重或引入采样均衡。

多头边界现状：

- 当前模型类暴露共享 encoder + `turn_head` / `response_head` 的接口，但训练入口仍是单任务 `turn` / `response`，尚未实现 `task: multitask` 的 batch 混合与 loss 聚合。
- 本轮 response 优化只改变 response 训练集过滤、response 指标和 response checkpoint 选择标准；不会改变 turn 配置、turn 数据集、turn loss 或 turn checkpoint 选择。

下一阶段优先补：

- 更大数据量的正式训练配置
- 更长训练轮数与更可靠的 eval report 留存
- 训练过程中的动作分布、候选规模和非法标签质量闸门
- 类别不均衡下的自动权重估计、采样均衡和少数类错例聚合
- 更高吞吐的数据格式后续方向：当前已经完成 state/action 特征离线预编码，下一步应根据 `data_wait`、GPU 利用率和内存占用决定是否继续推进 mmap/Arrow streaming 数据集，而不是回到训练期规则特征重算。
- 根据错例聚合决定是否回补规则

## 5. 阶段结论

如果按原始路线图划分，当前可以认为：

- `Task 1` 项目骨架初始化：已完成
- `Task 2` 牌表示与基础规则：已完成
- `Task 3` 天凤解析与事件 schema：已完成
- `Task 4` 回放与合法动作生成：已完成第一版
- `Task 5` 样本构建：已完成第一版
- `preprocess / validate-events`：已完成第一版
- `Task 6` baseline 模型与训练引擎：已完成 `turn/response` 第一版
- `Task 7` 离线评估：已完成第一版
- `Task 8` 多任务训练：未开始
- `Task 9` 训练机适配：只完成配置骨架，未验证训练链路

## 6. 建议的下一步任务

建议按下面顺序推进：训练先行，规则边界按证据回补。

### 任务 1：扩大数据构建与质量闸门

目标：

- 在更多真实牌谱上跑完整数据链路，确认当前规则集对训练数据的污染率是否可接受。

建议范围：

1. 扩大输入牌谱规模，跑通 `preprocess / validate-events / build-dataset --workers N / make-manifest`
2. 汇总构建失败率、非法标签率、动作分布、候选规模分布和 split 样本数
3. 对失败样本只保留轻量定位信息，依赖 `inspect-sample` 和 `dump-case` 回溯上下文

最低验收标准：

- 更大数据集能稳定产出 `turn/response` manifest。
- 非法标签率和构建失败率有明确统计。
- 发现的高频失败能定位到具体 `game_id / kyoku_index / step_index`。

### 任务 2：深入训练当前 `turn/response` baseline

建议范围：

1. 增大训练数据量、epoch、hidden size 和 batch 配置
2. 分别训练 `turn` 与 `response`，固定输出 `train_metrics.json`、checkpoint 和 eval report
3. 对比不同配置的 loss、top-k accuracy、per-kind 指标和候选规模分桶
4. 记录可复现实验配置，避免只看单次 smoke 结果

### 任务 3：规则边界按训练反馈回补

建议方向：

- 更完整的和牌机会放弃语义与残余振听边界
- 其他途中流局边界
- 更严格的立直限制
- 多家荣和等边界情况

触发条件：

- 某类规则缺口在大规模构建中导致高频失败或非法标签。
- 某类规则缺口在 eval 错例中形成稳定、可复现的主要误差来源。
