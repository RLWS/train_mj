# 项目进度状态

更新时间：`2026-05-12`

## 1. 总体判断

当前项目已经明显超过“纯规划阶段”，可以视为：

- `Phase 0` 第一版已完成：规则与数据的最小闭环已经跑通。
- `Phase 1` 已推进到大数据单任务 baseline、第一轮特征复训和首轮多任务 baseline 之后：`turn/response` 都已经有 P800 baseline 与 improved 两轮 valid/test eval，v1.1 multitask 也已经有首轮 test eval 和可对照指标。

换句话说，仓库当前已经从“只能产样本”进入“能用大数据训练、归档 run、做阈值校准和分析模型取舍”的阶段。v1.0.0 已固定为 2026-05-10 P800 improved 单任务 baseline；v1.1 首轮多任务结果显示 `turn` 明确受益，但 `response` 经 raw eval 与 valid/test calibration 复核后仍不能替代 improved 单任务，后续重点转为调度/权重和少数类质量实验。DDP 和系统化 serving 仍未完成。

## 2. 已验证结果

以下结果已经在仓库内实际验证：

- `pytest` 全量通过，当前结果为 `101 passed`；本轮新增 turn 牌河时序 V1 特征后，`tests/test_action_features.py`、`tests/test_dataset_writer.py`、`tests/test_train_smoke.py` 和全量 `pytest` 已通过。
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
- 离线 `eval` 现在默认使用 streaming dataset：启动时只读取 manifest 与 shard 元信息，迭代时按 parquet shard 读取预编码特征和错例报告需要的动作 JSON，不再把整份 valid/test split 常驻内存。`eval.num_workers` 可单独配置；未配置时沿用 `train.valid_num_workers`。
- 对最小 `turn` 与 `response` smoke checkpoint 实测，可输出：
  - `loss`
  - `top1 / top3 / top5 accuracy`
  - 按动作类别聚合的样本数与 `top-k` 指标
  - 按候选动作数量分桶的 `top-k` 指标
  - 首批错例样本的 label、top-k 预测、legal actions，以及可直接执行的 `inspect-sample` / `dump-case` 跳转命令
  - JSON 评估报告
- response 评估报告已补充 response 专用诊断：`single_candidate` / `multi_candidate`、`pass` vs `non_pass` precision / recall / F1、`false_call_rate`、`missed_call_rate`、候选集合聚合与 `label->pred` confusion。
- `python -m mjai.cli calibrate-response --ckpt <response_ckpt> --split valid --manifest <manifest> --report <report>` 可运行；它会扫描 pass/call score margin，并为 `false_call_rate <= 0.010 / 0.008 / 0.005` 等目标选择 `non_pass_f1` 最好的 operating point。
- `python -m mjai.cli sample-response-errors --ckpt <response_ckpt> --split valid --manifest <manifest> --output <jsonl>` 可运行；它会按 `label->pred` confusion 分桶做 reservoir sampling，默认目标为 `pass->chi / pass->pon / chi->pass / pon->pass`，`--seed` 默认继承 checkpoint 全局 `seed`。
- `python -m mjai.cli sample-turn-errors --ckpt <turn_ckpt> --split valid --manifest <manifest> --output <jsonl>` 可运行；它会按 label kind 与 label rank 分桶做 reservoir sampling，默认抽 `discard@rank2` 与 `discard@rank3`，用于 turn top-k 错例分析。
- 2026-05-03 response action feature 增强后复训已完成并记录在 [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md)：训练过滤后保留 `141423 / 1490034` 条 response 训练样本；最新 `best.pt` 在 valid/test 上分别达到 `multi_candidate top1=0.8521/0.8502`、`multi_candidate macro top1=0.7882/0.8697`、`non_pass_f1=0.7215/0.7189`、`false_call_rate=0.0111/0.0111`、`missed_call_rate=0.1425/0.1466`。
- 2026-05-06 P800 大数据单任务 baseline 已完成，训练产物已归档到 `runs/2026-05-06_p800_baseline/`，分析记录见 [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md) 第 10 节。
  - `turn_p800/best.pt`：train `83077288` 条、valid `10404270` 条、test `10393095` 条；test `top1=0.7008`、`top3=0.9446`、`top5=0.9853`、`macro_top1=0.9134`，其中 `discard top1=0.6958`、`riichi_discard top1=0.8534`。
  - `response_p800/best.pt`：原始 train `247177362` 条，过滤 pass-only 后有效 train `23120538` 条；test `multi_candidate top1=0.8619`、`multi_candidate macro top1=0.8742`、`non_pass_f1=0.7131`、`false_call_rate=0.0118`、`missed_call_rate=0.0644`、`chi top1=0.8376`、`pon top1=0.9528`。
  - 本轮 P800 数据版本已归档到 `runs/_datasets/tenhou_20w_2026-05-06/`：`preprocess_summary.json` 记录 `212163` 个 game、`215224122` 条事件；`event_validation_20w.json` 记录 `files_seen=212163`、`files_valid=212163`、`files_failed=0`、`error_counts={}`；`build_summary.json` 记录 `103874653` 条 turn、`309055971` 条 response；`manifest_summary.json` 与 `train/valid/test.jsonl` 逐行累加结果完全一致，并和 P800 训练日志样本数对齐。
- 2026-05-10 P800 improved 复训已完成，训练产物已归档到 `runs/2026-05-10_p800_improved/`，分析记录见 [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md) 第 11 节。
  - `turn` 明确进步：test `top1=0.7075`、`top3=0.9493`、`top5=0.9868`，相对 baseline 分别提升 `+0.67pp / +0.48pp / +0.15pp`；`discard top1=0.7024` 提升 `+0.66pp`，`riichi_discard top1=0.8713` 提升 `+1.78pp`。副作用是 `ankan top1` 从 `0.9602` 降到 `0.8988`，需要错例排查。
  - `response` 是更保守的候选：test `multi_candidate top1=0.8686`、`multi_candidate macro top1=0.8843`、`non_pass_f1=0.7214`、`false_call_rate=0.0111`，均优于 baseline；但 `missed_call_rate` 从 `0.0644` 升到 `0.0728`，`chi top1` 从 `0.8376` 降到 `0.8198`，`pon top1` 从 `0.9528` 降到 `0.9475`。
  - response calibration 已完成 valid/test。improved 在同一 `pass_bias` 下误叫更少但漏叫更多；baseline 的默认候选 `pass_bias=0.4` 不直接迁移到 improved。v1.0.0 默认 response operating point 定为 `pass_bias=0.3`：test `false_call_rate=0.925%`、`missed_call_rate=9.29%`、`non_pass_f1=74.18%`、`multi_candidate top1=88.40%`、`chi/pon top1=78.4%/93.1%`。
- 2026-05-12 P800 multitask v1.1 首轮训练已完成，训练产物已归档到 `runs/2026-05-12_p800_multitask_v1_1/`，分析记录见 [multitask-training.md](multitask-training.md) 第 7 节和 run 目录下的 `analysis.md`。
  - `turn` 明确优于 2026-05-10 improved：test `top1=0.7169`、`top3=0.9538`、`top5=0.9884`，相对 improved 分别提升 `+0.94pp / +0.45pp / +0.16pp`；`discard top1=0.7118`、`riichi_discard top1=0.8803`、`non_discard top1=0.9207` 均提升，`ankan top1` 也从 improved 的 `0.8988` 恢复到 `0.9186`。
  - `response` 仍不能替代 improved 单任务：补跑 `best_response.pt` 后，raw test `multi_candidate top1=0.8646`、`multi_candidate macro top1=0.8518`、`non_pass_f1=0.7149`、`false_call_rate=0.0114`，比 multitask `best.pt` 更接近 improved，但仍弱于 improved 的 `0.8686 / 0.8843 / 0.7214 / 0.0111`。valid/test calibration 也未反转结论：test `pass_bias=0.3` 时 multitask best_response 为 `false_call=0.933%`、`missed=9.91%`、`non_pass_f1=73.73%`，弱于 improved 的 `0.925% / 9.29% / 74.18%`；test `pass_bias=0.4` 时 false call 基本持平，但 missed call 更高、F1 更低，且 `daiminkan` 明显偏低。不应把 multitask response 作为默认候选。
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
  - 2026-05-09 已增强 turn discard 候选特征：`TURN_ACTION_DIM` 从 `69` 增至 `85`，新增候选弃牌在自家/他家牌河中的出现次数、早巡/最近 6 巡现物、多家现物和相对座位现物信号，用于改善 rank 2/3 discard 的 top1 排序。
  - response 候选动作特征已补充第一版 call 收益/代价信息：叫牌前后向听、有效牌数量、有效牌枚数、赤牌/dora 消耗、对子/搭子消耗、役牌碰、chi 形状、巡目和剩余巡目。
  - 2026-05-09 已继续增强 response call/context 特征：`RESPONSE_ACTION_DIM` 从 `77` 增至 `91`，新增可见牌修正有效牌枚数、开副露后明显役路启发式、无役风险、七对子/国士等门清特殊路线破坏信号；样本状态同步写出 `all_melds`，用于扣除其他玩家副露里的可见牌。该特征是启发式，不等价于完整番种判定。
- `mjai/models/`
  - 已有最小共享编码器与 `turn/response` 候选打分头。
- `mjai/training/`
  - 已有最小训练循环、`turn/response` 标签样本加权 loss、总体与宏平均 accuracy、按动作类别和主类/非主类分组的训练指标、checkpoint 落盘。
  - response 训练已支持 `train.response_min_train_candidate_count`，当前 response smoke / P800 配置默认用 `2` 过滤训练集中的 pass-only 单候选样本；valid/test 评估仍保留全量样本。
  - response 的 `best.pt` / `best_response.pt` 现在按 `non_pass_f1 - false_call_rate` 选择，避免只被 overall accuracy 或 pass-only 样本支配；`best_accuracy.pt`、`best_loss.pt`、`best_macro.pt` 仍保留。
  - 2026-05-03 action feature 增强后复训显示 response 已经达到进入更大数据 baseline 的门槛：false call 与 missed call 同时下降，`chi/pon` top1 也有明确提升。
  - 2026-05-06 P800 大数据 baseline 进一步确认当前配置有效：response 的 `chi/pon` exact action 明显提升，missed call 大幅下降；下一步重点转为 pass/call 校准和 hard negative 错例分析，而不是继续盲目调权重。
  - 已支持 `pin_memory`、`persistent_workers`、`prefetch_factor` 和 CUDA `non_blocking` 传输；Windows 本地 smoke 配置默认 `num_workers: 0`，避免 spawn worker 复制整份内存数据集。
  - 训练循环已接入 `train.precision` 的 CUDA/XPU autocast；P800 配置默认 `bf16`，CUDA 路径默认开启 TF32，并在启动日志中打印实际精度设置。
  - 训练指标聚合已减少 batch 内逐样本 `.item()` 同步：每个 batch 将 `correct`、per-sample loss、response rank/prediction 一次性搬到 CPU 后聚合，降低小 MLP ranker 训练时的 GPU 同步碎片。
  - response 训练 split 默认关闭完整 response 诊断，跳过 `legal_actions_json` 读取/解析和 label-rank/candidate-set 统计；valid/test 诊断仍保留，用于 `best_response.pt` 选择和离线评估。
  - train split 的按标签/主类分组指标已改为 `label_kind_ids` + `torch.bincount` 批量聚合，避免 `batch_size=8192/16384` 时 Python 逐样本循环拖慢训练。
  - 已支持由 `train.log_every_n_steps` 控制的 batch 进度日志，输出 batch 进度、`batch_time`、`data_wait`、`compute_time`、窗口平均耗时、samples/s、窗口 loss/accuracy 和 ETA。
- `mjai/eval/`
  - 已有第一版离线评估入口，可基于 checkpoint 复用训练配置做 split 级评估与 JSON 报告导出。
  - 已支持按动作类别、主类/非主类分组、候选规模分桶与首批错例导出。
  - response checkpoint 额外输出 `candidate_count > 1` 子集、pass/non-pass 二分类、false call / missed call 和候选集合维度指标；turn checkpoint 的 `response` 诊断字段为 `null`。
  - 已新增 response pass/call 校准扫描：固定 checkpoint，在一组 `pass_bias` 下输出 false call、missed call、non-pass precision/recall/F1、multi-candidate top1 和按动作类别 top1，并按 false call 目标挑选 operating point。
  - 已新增 response 错例抽样命令：重新跑 manifest 推理后按 `label->pred` confusion 抽样，避免误用 `eval.misclassified_examples` 里默认最多 `10` 条的调试样本。
  - 已新增 turn top-k 错例抽样命令：重新跑 manifest 推理后按 label kind 与 label rank 抽样，默认目标为 `discard@rank2` 与 `discard@rank3`。
- `tests/`
  - 规则、解析、样本构建、CLI、配置和数据写出都有测试覆盖。
  - `turn/response` smoke train、offline eval、response/turn 错例抽样、`inspect-sample` 与 `dump-case` 都有回归测试。

### 部分完成

- `mjai/cli.py`
  - 子命令框架完整。
  - `preprocess`、`validate-events`、`build-dataset`、`make-manifest`、`train`、`eval`、`calibrate-response`、`sample-response-errors`、`sample-turn-errors`、`inspect-sample`、`dump-case` 已可用。
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

- v1.1 已新增 `MultitaskMahjongPolicyNet`：`turn/response` 各自保留 state projection、action encoder 和 head，共享一段状态 encoder 主干，解决两个任务输入维度不同但仍需共享表征的问题。
- 训练入口已支持 `task: multitask`。v1.1 使用 epoch 级顺序调度；当前代码已新增 `batch_interleave`，会按有效训练 batch 进度比例交错更新 `turn` / `response`，valid 阶段仍分别评估两路指标。由于 response train 会过滤 pass-only 样本，P800 v1.2 显式配置 `task.interleave_batches`，避免用 response 原始行数高估调度权重。
- 多任务 checkpoint 可通过 `python -m mjai.cli eval --ckpt <ckpt> --task turn|response` 分别评估，也已兼容 `calibrate-response`、`sample-response-errors` 和 `sample-turn-errors`。
- 首轮 P800 多任务训练已经完成：`turn` 收益明确，`response` 在 `best_response.pt` raw test 与 valid/test calibration 下仍弱于 improved。v1.2 已将默认多任务调度切到 `batch_interleave`，用于验证 epoch-sequential 覆盖效应；自动 loss scale、value head 与 DDP 仍暂不混入本轮变量。

下一阶段优先补：

- 更大数据量的正式训练配置
- 更长训练轮数与更可靠的 eval report 留存
- 训练过程中的动作分布、候选规模和非法标签质量闸门
- 类别不均衡下的自动权重估计、采样均衡和少数类错例聚合
- 更高吞吐的数据格式后续方向：当前已经完成 state/action 特征离线预编码，下一步应根据 `data_wait`、GPU 利用率和内存占用决定是否继续推进 mmap/Arrow streaming 数据集，而不是回到训练期规则特征重算。
- 根据错例聚合决定是否回补规则

数据重建与 P800 大数据训练提醒：

- turn/response action feature 维度已更新到当前数据版本，`runs/_datasets/tenhou_20w_2026-05-06/` 与 improved run 的样本数和训练日志对齐。后续只要 feature schema、parser/rules 或 split 逻辑变化，就必须新建 dataset version 并重跑 `build-dataset` / `make-manifest`。
- 20 万局量级的 baseline、improved 单任务训练和首轮 v1.1 多任务训练都已跑完，valid/test 指标稳定。multitask response test calibration 已完成；下一轮主线是跑 v1.2 `batch_interleave` 调度并和 v1.1 对照，之后再判断是否需要调整 response loss scale 或少数类处理。不要同时混入 DDP、value head 和复杂模型。
- 当前缺口不是“能否训练”：后续应继续按 `runs/<date>_<exp>/` 固化每轮模型证据，并通过 `dataset_id` 引用 `runs/_datasets/<dataset_id>/` 中的数据证据。
- v1.0.0 response 默认 operating point 已定为 `pass_bias=0.3`；`0.16` 作为偏召回候选，`0.4` 作为保守候选。后续推理/服务配置应显式记录该值。

## 5. 阶段结论

如果按原始路线图划分，当前可以认为：

- `Task 1` 项目骨架初始化：已完成
- `Task 2` 牌表示与基础规则：已完成
- `Task 3` 天凤解析与事件 schema：已完成
- `Task 4` 回放与合法动作生成：已完成第一版
- `Task 5` 样本构建：已完成第一版
- `preprocess / validate-events`：已完成第一版
- `Task 6` baseline 模型与训练引擎：已完成 `turn/response` 第一版，并已跑通 P800 大数据单任务 baseline
- `Task 7` 离线评估：已完成第一版
- `Task 8` 多任务训练：已完成可训练 baseline 和 P800 首轮验证；`turn` 正收益已确认，`response` 需要调度/权重和少数类质量改进
- `Task 9` 训练机适配：P800 单卡 baseline 已验证；8 卡 DDP 仍未实现/验证

## 6. 建议的下一步任务

建议按下面顺序推进：训练先行，规则边界按证据回补。

### 任务 1：接入 response operating point

目标：

- 将 v1.0.0 response 默认 `pass_bias=0.3` 接入后续推理/服务配置，并保持 raw、偏召回、默认、保守四个评估口径。

建议范围：

1. 默认值：`pass_bias=0.3`
2. 偏召回候选：`pass_bias=0.16`
3. 保守候选：`pass_bias=0.4`
4. 后续多任务或新特征实验必须同时报告 raw 与默认 calibrated 指标

最低验收标准：

- valid/test 曲线一致。
- raw、偏召回、默认和保守点都有 false call、missed call、non-pass F1、multi top1、`chi/pon` top1 记录。
- 推理默认值有明确配置位置或设计记录。

### 任务 2：response 定向错例抽样

建议范围：

1. 对 improved response 按 raw 和选定默认 bias 各抽一份 `pass->chi`、`pass->pon`、`chi->pass`、`pon->pass`
2. `pass->chi/pass->pon` 重点判断误叫是否来自防守/点况信息缺失、无役风险、玩家风格差异或 action feature 不足
3. `chi->pass/pon->pass` 重点判断 improved 变保守后漏掉了哪些收益明确的副露
4. 如果后续命令支持 exact-action confusion，应增加 `chi->wrong_chi` 与 `pon->wrong_pon`，解释本轮 `chi/pon` top1 回退

### 任务 3：turn 少数类回退排查

建议范围：

1. 以 improved turn 为当前主线 checkpoint，确认 `discard@rank2/rank3` 错例是否少了安全性/近期现物相关混淆
2. 针对 `ankan` 和 `kakan` 抽错例，确认回退是否由牌河安全性特征干扰、少数类样本波动或 class weight 不合适导致
3. 如果杠类回退是真实问题，优先考虑杠类独立特征或权重微调，不要为了少数类直接回滚 discard 已验证有效的特征

### 任务 4：继续规范 run 归档与质量闸门

建议范围：

1. 后续统一使用 `runs/<date>_<exp>/` 保存配置、日志、checkpoint 索引、eval report、calibration report 和错例样本
2. 数据集级证据统一放到 `runs/_datasets/<dataset_id>/`，run 通过 `dataset_id` 引用它
3. 固化非法标签复核结果，至少记录 `label_action not in legal_actions` 的 turn/response 计数
4. 在 run summary 中记录最终结论、默认 operating point、主要收益和已知回退

### 任务 5：规则边界按训练反馈回补

建议方向：

- 更完整的和牌机会放弃语义与残余振听边界
- 其他途中流局边界
- 更严格的立直限制
- 多家荣和等边界情况

触发条件：

- 某类规则缺口在大规模构建中导致高频失败或非法标签。
- 某类规则缺口在 eval 错例中形成稳定、可复现的主要误差来源。
