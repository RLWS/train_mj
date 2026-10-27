# Smoke Train 与调试上手教程

> 2026-10-27 实验分支更新：`codex/full-state-features` 使用 `full_state_v2`，turn/response state 为 11969/12012 维，候选维度仍为 89/91。旧模型与数据继续使用原 checkout；本分支必须从原始 XML 重建并重训。新训练命令、API 字段和验证证据见 [完整状态训练流程](full-state-training.md)，特征布局见 [模型输入](model-inputs.md)。下文旧 run 的结果属于历史 baseline，不代表 v2 牌力。


更新时间：`2026-07-31`

这份教程带你完整走一遍当前仓库已经打通的最小闭环，目标不是只把命令跑通，而是让你熟悉现在项目里这几类对象之间的关系：

- `bronze_events`：事件分片
- `silver_samples`：监督学习样本
- `manifests`：训练/验证切分清单
- `turn` / `response` smoke train
- `inspect-sample`：看单条样本
- `dump-case`：回到原始事件上下文

适用目录：仓库根目录。

## 1. 你会得到什么

跑完这篇之后，你应该能做到：

1. 用仓库自带夹具生成一套最小 smoke 数据。
2. 先校验 `bronze_events`，再构建 `turn/response` 样本。
3. 分别训练 `turn` 和 `response` 两个最小 baseline。
4. 找到训练输出和评估输入的位置。
5. 用 `inspect-sample` 抽查一条样本，读懂 `sample_id`。
6. 用 `dump-case` 从 `sample_id` 反查到原始事件上下文。

## 2. 前置条件

确保当前终端位于仓库根目录：

```powershell
cd <repo-root>
```

当前流程依赖 Python `3.10+` 和可用的 PyTorch 环境。项目的 editable install 不会安装 PyTorch；请先按本机 CUDA/P800 vendor 环境安装匹配的 PyTorch，再安装项目依赖：

```powershell
pip install -e ".[dev,training,api]"
```

## 3. 第一步：准备 smoke 数据

直接使用仓库自带的 Tenhou XML 夹具：

```powershell
python -m mjai.cli preprocess --input tests/fixtures/tenhou --output data/bronze_events
python -m mjai.cli validate-events --input data/bronze_events --report outputs/reports/smoke_event_validation.json
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 1 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
```

P800/Linux 上如果要从 `/workspace/dataset` 一步跑完整链路，可以直接用仓库根目录的 [start.sh](../start.sh)：

```bash
./start.sh
```

脚本默认执行：

```text
preprocess -> validate-events -> build-dataset -> make-manifest -> train turn/response -> eval turn/response
```

其中 `/workspace/dataset` 只作为原始 XML 输入目录，`bronze_events`、`silver_samples`、`manifests`、reports、checkpoints 和 logs 仍写到当前仓库的 `data/` 与 `outputs/` 下。常用开关如下：

```bash
TASKS=turn ./start.sh
TASKS=response ./start.sh
CLEAN_DATA=1 RUN_CALIBRATION=1 ./start.sh
RUN_VALIDATE=0 ./start.sh
RUN_TRAIN=0 RUN_EVAL=0 ./start.sh
RUN_PREPROCESS=0 RUN_VALIDATE=0 SKIP_BUILD_VALIDATION=1 ./start.sh
```

`TASKS` 只影响训练和对应 eval，默认 `turn response` 都运行。`RUN_VALIDATE=0` 会跳过独立 validation report；如果同时保持 `SKIP_BUILD_VALIDATION=0`，`build-dataset` 仍会在构建样本前做内部回放校验。
`CLEAN_DATA=1` 会在仓库内部清理旧的 `data/bronze_events`、`data/silver_samples` 和 `data/manifests`，并要求同时开启 `RUN_PREPROCESS=1`、`RUN_BUILD_DATASET=1`、`RUN_MANIFEST=1`；它适合本轮 `round_wind` 这类不兼容旧数据的全量重建。`RUN_CALIBRATION=1` 会在训练后跑 response valid/test calibration，默认扫描 `CALIBRATION_PASS_BIASES` 和 `CALIBRATION_NO_YAKU_CALL_PENALTIES=0,0.2,0.4`。

这里推荐先单独跑 `validate-events`，确认 `files_failed=0` 后，再在 `build-dataset` 使用 `--skip-validation`，避免同一批事件在构建样本时重复回放校验。`build-dataset` 现在会输出 `elapsed_s`、`games_per_s` 和 `samples_per_s`，可以直接用来判断构建吞吐。

当前 `silver_samples` parquet 会写入离线预编码的 `state_features_bin` 和 `candidate_features_bin`。`turn` 的 discard hand-shape 特征，包括向听数、有效牌数量等 CPU 密集计算，已经从训练期 DataLoader 移到 `build-dataset` 阶段。更新到这个 schema 后，旧的 `data/silver_samples` 需要删除或覆盖后重新生成，并重新执行 `make-manifest`。这次还新增了四家立直状态、对手立直巡目和全局牌河顺序 token，所以旧 checkpoint 也不能直接复用。

2026-06-01 起，Tenhou `INIT seed` 会在 `preprocess` 阶段解析为 `round_wind`，用于修正 response 场风役牌路线和无役风险判断。旧 `data/bronze_events` 不再兼容；更新后必须从 `preprocess` 开始重新生成 `bronze_events`、`silver_samples` 和 manifest。

2026-07-31 对当前 21 个 XML 夹具实测得到 `7386` 条事件、`3532` 条 turn 样本、`10506` 条 response 样本，事件校验 `21/21` 通过，manifest game 数为 `19/1/1`。

执行完成后会得到：

- 事件分片目录：`data/bronze_events`
- 事件校验报告：`outputs/reports/smoke_event_validation.json`
- 样本分片目录：`data/silver_samples`
- manifest 目录：`data/manifests`

最关键的文件是：

- `data/manifests/train.jsonl`
- `data/manifests/valid.jsonl`
- `data/manifests/test.jsonl`

这一步对应的含义是：

- `preprocess`：XML -> 标准化事件
- `validate-events`：回放校验事件分片
- `build-dataset`：事件 -> `turn/response` 样本
- `make-manifest`：样本分片 -> 训练/验证/测试切分

如果你没有先跑 `validate-events`，不要加 `--skip-validation`：

```powershell
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 1
```

在 P800 上构建大规模真实牌谱时，建议从下面这种命令起步，再根据 `samples_per_s` 调整 worker 数：

```bash
python -m mjai.cli validate-events --input data/bronze_events --report outputs/reports/event_validation.json
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 32 --skip-validation
```

`build-dataset` 现在优先使用进程池并行，适合 CPU 密集的规则回放和合法动作生成。可以试 `--workers 32`、`48`、`64`，取 `samples_per_s` 最高且机器负载稳定的值。

因为候选特征已经离线预编码，训练阶段不再需要用很多 DataLoader worker 做特征计算。Windows 本地配置默认 `num_workers: 0`，主要避免 spawn/进程通信开销和平台稳定性问题；默认训练主线使用 streaming dataset，并不会在初始化时把整份 manifest 样本 JSON 全部常驻内存。Linux/P800 上可以从较小的 worker 数起步，再看训练日志里的 `data_wait` 是否仍然明显。

P800/CUDA 单卡训练现在会读取 `train.precision`、`train.allow_tf32` 和 `train.float32_matmul_precision`。`turn_p800.yaml` 与 `response_p800.yaml` 默认开启 `precision: bf16`、`allow_tf32: true` 和 `float32_matmul_precision: high`；当前 batch size 分别为 turn `16384`、response `32768`。训练启动日志会打印 `precision`、`autocast`、`autocast_dtype`、`allow_tf32`。如果 `data_wait` 接近 0 但 GPU 利用率仍低，瓶颈通常不是 DataLoader，而是当前 MLP ranker 单 step 计算量偏小；继续加 worker 通常收益不大。

`response` 训练还要额外注意诊断开销。训练 split 默认 `response_train_diagnostics: false`，不会读取/解析 `legal_actions_json`，也不会逐样本计算 response candidate-set 诊断；valid split 默认 `response_eval_diagnostics: true`，仍会输出 `non_pass_f1`、false/missed call 等指标，并继续用于选择 `best_response.pt`。如果为了排查训练集诊断临时打开 `response_train_diagnostics: true`，GPU 利用率会明显下降。

`response` 的候选数通常远少于 `turn`，即使 DataLoader 不阻塞，单个 batch 的 GPU 计算也更轻，因此当前 response batch size 高于 turn。训练 split 的类别指标使用批量聚合，避免大 batch 下 Python 逐样本循环成为瓶颈。

## 4. 第二步：跑 turn smoke train

`configs/turn_local_smoke.yaml` 当前是 `28` 个 epoch，不是一次极短的一轮测试；完整运行时间取决于设备。只想验证链路时，可以复制配置到本地忽略文件并把 `train.epochs` 改为 `1`，不要改动共享基线配置。

命令：

```powershell
python -m mjai.cli train --config configs/turn_local_smoke.yaml
```

配置文件：

- [turn_local_smoke.yaml](../configs/turn_local_smoke.yaml)

这个配置默认读取：

- `data/manifests/train.jsonl`
- `data/manifests/valid.jsonl`

默认输出位置：

- checkpoint：`outputs/checkpoints/turn_local_smoke`
- log：`outputs/logs/turn_local_smoke`

重点看这些文件：

- `config.resolved.yaml`
- `train_metrics.json`
- `best.pt`
- `best_accuracy.pt`
- `best_loss.pt`
- `best_macro.pt`
- `final.pt`
- `epoch_metrics.jsonl`

## 5. 第三步：跑 response smoke train

`configs/response_local_smoke.yaml` 当前是 `10` 个 epoch；只验证链路时同样建议用本地一轮覆盖配置。

命令：

```powershell
python -m mjai.cli train --config configs/response_local_smoke.yaml
```

配置文件：

- [response_local_smoke.yaml](../configs/response_local_smoke.yaml)

默认输出位置：

- checkpoint：`outputs/checkpoints/response_local_smoke`
- log：`outputs/logs/response_local_smoke`

和 `turn` 一样，至少确认这些文件存在：

- `config.resolved.yaml`
- `train_metrics.json`
- `best.pt`
- `best_accuracy.pt`
- `best_loss.pt`
- `best_macro.pt`
- `final.pt`
- `epoch_metrics.jsonl`

## 6. 第四步：先抽查一条样本

现在不要急着看模型结果，先看一条原始监督样本长什么样：

```powershell
python -m mjai.cli inspect-sample --dataset data/silver_samples --index 0
```

你会看到一段 JSON，关键字段包括：

- `sample_id`
- `source_path`
- `source_row_index`
- `decision_type`
- `candidate_count`
- `label_action`
- `legal_actions`
- `state`

当前第一条样本大致会长这样：

```json
{
  "sample_id": "2023010100gm-00e1-0000-8a80056a:0:2:0:turn",
  "game_id": "2023010100gm-00e1-0000-8a80056a",
  "kyoku_index": 0,
  "step_index": 2,
  "actor": 0,
  "decision_type": "turn"
}
```

这里最关键的是 `sample_id` 的结构：

```text
game_id : kyoku_index : step_index : actor : decision_type
```

也就是说：

- `2023010100gm-00e1-0000-8a80056a`：这一局牌的唯一标识
- `0`：第几局
- `2`：这一局里的第几个事件 step
- `0`：当前玩家座位
- `turn`：这是自家回合决策样本

`inspect-sample` 的用途是回答两个问题：

1. 当前样本的监督标签是什么？
2. 当前样本的合法候选动作集合是什么？

这是之后理解训练和错例分析的基础。

## 7. 第五步：用 dump-case 回到原始事件

拿上一步的 `sample_id`，你就能把它还原到 `bronze_events` 里的原始事件上下文。

对于上面的样本：

```text
sample_id = 2023010100gm-00e1-0000-8a80056a:0:2:0:turn
```

可以直接拆成：

- `game_id = 2023010100gm-00e1-0000-8a80056a`
- `kyoku_id = 0`
- `step_id = 2`

然后执行：

```powershell
python -m mjai.cli dump-case --game-id 2023010100gm-00e1-0000-8a80056a --kyoku-id 0 --step-id 2
```

你会得到一段更靠近原始牌谱的 JSON，重点字段包括：

- `target_event`
- `kyoku_events`
- `context_before`
- `context_after`
- `source_path`

这里的含义是：

- `target_event`：你指定的那个事件
- `kyoku_events`：这一局里的完整事件序列
- `context_before` / `context_after`：目标事件前后的局部窗口
- `source_path`：这一局对应的 `bronze_events/<game_id>.parquet`

如果你想看更短的局部上下文，可以加 `--context-radius`：

```powershell
python -m mjai.cli dump-case --game-id 2023010100gm-00e1-0000-8a80056a --kyoku-id 0 --step-id 2 --context-radius 2
```

## 8. inspect-sample 和 dump-case 的关系

这两个命令是当前项目里最重要的一条调试链：

```text
silver sample -> sample_id -> game_id / kyoku_index / step_index -> bronze event context
```

推荐你的理解顺序是：

1. 先用 `inspect-sample` 看样本长什么样。
2. 记住 `sample_id`。
3. 再用 `dump-case` 看它在原始事件流里对应什么决策点。

这样你就能同时看到：

- 训练视角：模型看见了什么 state，label 是什么
- 规则/回放视角：原始事件到底发生了什么

## 9. 如果你还想多走一步

虽然你这次重点是 `train / inspect / dump-case`，但建议顺手也把 `eval` 走一遍，这样对当前项目更完整：

### turn eval

```powershell
python -m mjai.cli eval --ckpt outputs/checkpoints/turn_local_smoke/best_accuracy.pt --split valid --report outputs/checkpoints/turn_local_smoke/eval_valid_accuracy.json
```

### response eval

```powershell
python -m mjai.cli eval --ckpt outputs/checkpoints/response_local_smoke/best.pt --split valid --report outputs/checkpoints/response_local_smoke/eval_valid.json
```

当前 `eval` 报告里已经有：

- `loss`
- `top1 / top3 / top5 accuracy`
- `per_kind`
- `candidate_buckets`
- `misclassified_examples`

response checkpoint 还会额外输出 `response` 诊断块，重点看：

- `multi_candidate`：排除 pass-only 单候选后的 top-k / macro top-k。
- `binary.non_pass_precision / non_pass_recall / non_pass_f1`。
- `binary.false_call_rate`：真实 pass 但模型选择 chi/pon/ron 等响应的比例。
- `binary.missed_call_rate`：真实 chi/pon/ron 等响应但模型选择 pass 的比例。
- `candidate_sets` 与 `confusion`：按候选集合和 `label->pred` 聚合错法。

`eval` 默认按 manifest 中的 parquet shard 流式读取 valid/test split，不再在启动阶段把整份评估集加载到内存。报告能力保持不变：为了生成错例、top predictions 和 response 诊断，评估仍会读取 `legal_actions_json` 与 `label_action_json`，但不会读取更大的 `state_json`；需要看完整状态时继续用报告里的 `inspect-sample` 命令回查。大数据评估可以在配置里设置 `eval.num_workers`、`eval.prefetch_factor`、`eval.persistent_workers` 和 `eval.pin_memory`；未设置 `eval.num_workers` 时会沿用 `train.valid_num_workers`。

其中 `misclassified_examples` 现在已经带有：

- 样本定位信息：`sample_id / game_id / kyoku_index / step_index / actor`
- 可直接执行的 `inspect-sample --sample-id ...`
- 可直接执行的 `dump-case --bronze-dir ... --game-id ... --kyoku-id ... --step-id ...`

也就是说，评估报告里的错例已经可以直接跳回样本和原始事件上下文。但 `misclassified_examples` 只是调试定位样本，默认最多 `10` 条，并且按扫描顺序截断；不要把它当成全量错例池。

### response pass/call 校准

如果 response 的 false call 绝对量偏高，可以固定 checkpoint 做校准扫描，不需要重新训练：

```powershell
python -m mjai.cli calibrate-response --ckpt outputs/checkpoints/response_local_smoke/best.pt --split valid --manifest data/manifests/valid.jsonl --report outputs/checkpoints/response_local_smoke/response_calibration_valid.json --pass-biases 0,0.10,0.20,0.40,1.0,2.0 --no-yaku-call-penalties 0,0.4,0.8 --false-call-targets 1%,0.8%,0.5%
```

报告里的 `points` 是每个 `pass_bias` 与 `no_yaku_call_penalty` 组合的完整指标，`selections` 是每个 false call 目标下推荐的 operating point。除常规 false call / missed call 外，报告还会写入 `no_yaku_false_call_rate` 和 `no_yaku_false_call_share`，用于专门追踪 `features[38] == 1` 的无明显役路误叫；`no_yaku_equivalent_call_margin` 表示无役候选实际需要超过 pass 的等效 margin。这个命令只改变推理选择阈值和无役候选扣分，不会改 checkpoint 权重。

大规模 eval/calibration 可以在配置中设置 `eval.distributed: true`，再用 `torchrun` 启动。默认配置仍为单进程；分布式模式要求 `WORLD_SIZE > 1`，各 rank 分片读取数据，最终只由 rank 0 写报告。

### response 错例抽样

需要系统分析 `pass->chi`、`pass->pon`、`chi->pass`、`pon->pass` 这类 hard negative 时，用专门的抽样命令重新跑全量 manifest 推理：

```powershell
python -m mjai.cli sample-response-errors --ckpt outputs/checkpoints/response_local_smoke/best.pt --split valid --manifest data/manifests/valid.jsonl --output outputs/checkpoints/response_local_smoke/response_error_samples_valid.jsonl --targets "pass->chi,pass->pon,chi->pass,pon->pass" --per-target 100 --pass-bias 2.0 --no-yaku-call-penalty 0.8
```

`--seed` 默认继承 checkpoint 配置里的全局 `seed`；只有需要复现实验外的另一组随机样本时才显式传入。`--pass-bias 0 --no-yaku-call-penalty 0` 表示 raw argmax；40w 较高召回对照为 `0.0 + 0.4`；当前 API 极保守默认为 `2.0 + 0.8`，其历史 valid 扫描 missed call 约 `31.52%`。2026-05-10 improved 的 `0.3 + 0.0` 只适用于对应历史 checkpoint，不能把不同模型的阈值直接互换。PowerShell 会把未引用的 `>` 当成重定向符号，所以 `--targets` 必须加引号。

输出是 JSONL，每行一条样本，包含 `sample_id`、`confusion`、label/pred action、top predictions、legal actions，以及可直接执行的 `inspect-sample` 和 `dump-case` 命令。

### turn discard top-k 错例抽样

需要看 turn 模型“差一点排对”的 discard 样本时，用 `sample-turn-errors` 重新跑全量 manifest 推理，并按 label kind 与 label rank 分桶抽样：

```powershell
python -m mjai.cli sample-turn-errors --ckpt outputs/checkpoints/turn_local_smoke/best.pt --split valid --manifest data/manifests/valid.jsonl --output outputs/checkpoints/turn_local_smoke/turn_discard_rank23_valid.jsonl --label-kinds discard --ranks 2,3 --per-rank 100
```

`--label-kinds` 默认是 `discard`，`--ranks` 默认是 `2,3`，所以生产分析时通常可以省略这两个参数。输出是 JSONL，每行包含 `sample_id`、`target`（例如 `discard@rank2`）、`label_rank`、label/pred action、top predictions、legal actions，以及可直接执行的 `inspect-sample` 和 `dump-case` 命令。

## 10. 一套建议你自己手动跑的完整命令

如果你就是想完整熟悉当前项目，建议按下面顺序走一遍：

```powershell
python -m mjai.cli preprocess --input tests/fixtures/tenhou --output data/bronze_events
python -m mjai.cli validate-events --input data/bronze_events --report outputs/reports/smoke_event_validation.json
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 1 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests

python -m mjai.cli train --config configs/turn_local_smoke.yaml
python -m mjai.cli train --config configs/response_local_smoke.yaml

python -m mjai.cli inspect-sample --dataset data/silver_samples --index 0
python -m mjai.cli dump-case --game-id 2023010100gm-00e1-0000-8a80056a --kyoku-id 0 --step-id 2
python -m mjai.cli eval --ckpt outputs/checkpoints/response_local_smoke/best.pt --split valid --report outputs/checkpoints/response_local_smoke/eval_valid.json
python -m mjai.cli calibrate-response --ckpt outputs/checkpoints/response_local_smoke/best.pt --split valid --manifest data/manifests/valid.jsonl --report outputs/checkpoints/response_local_smoke/response_calibration_valid.json --no-yaku-call-penalties 0,0.4
python -m mjai.cli sample-response-errors --ckpt outputs/checkpoints/response_local_smoke/best.pt --split valid --manifest data/manifests/valid.jsonl --output outputs/checkpoints/response_local_smoke/response_error_samples_valid.jsonl --targets "pass->chi,pass->pon,chi->pass,pon->pass" --per-target 100 --pass-bias 2.0 --no-yaku-call-penalty 0.8
python -m mjai.cli sample-turn-errors --ckpt outputs/checkpoints/turn_local_smoke/best.pt --split valid --manifest data/manifests/valid.jsonl --output outputs/checkpoints/turn_local_smoke/turn_discard_rank23_valid.jsonl
```

`configs/response_local_smoke.yaml` 默认设置了 `train.response_min_train_candidate_count: 2`。response 训练会过滤 `candidate_count == 1` 的 pass-only 样本，valid/test eval 仍保留全量样本。

注意：turn 候选动作特征已经包含 discard 后向听、有效牌、牌河时序和按玩家拆开的现物信号；state 特征包含四家立直状态、对手立直巡目和全局牌河顺序 token。response 候选动作特征已经包含 call 前后向听、有效牌、赤牌/dora 消耗和 chi/pon 形状等离线预编码特征。修改这些特征或更新代码后，需要重新运行 `build-dataset` 和 `make-manifest`，否则旧 `state_features_bin` / `candidate_features_bin` 的维度或语义会和当前模型不匹配。涉及 `round_wind` 时还必须先重新运行 `preprocess`，不能复用旧 `bronze_events`。

如果你再多走一步，把评估也跑上：

```powershell
python -m mjai.cli eval --ckpt outputs/checkpoints/turn_local_smoke/best_accuracy.pt --split valid --report outputs/checkpoints/turn_local_smoke/eval_valid_accuracy.json
python -m mjai.cli eval --ckpt outputs/checkpoints/response_local_smoke/best.pt --split valid --report outputs/checkpoints/response_local_smoke/eval_valid.json
```

## 11. 你现在最值得建立的心智模型

当前仓库虽然已经有 train/eval/inspect/dump-case，但它还不是一个“完整麻将 AI 系统”，而是一个已经打通最小闭环的工程骨架。

你现在最应该熟悉的是这条链：

```text
Tenhou XML
-> bronze_events
-> validate-events
-> silver_samples
-> manifest
-> train turn/response
-> inspect-sample
-> dump-case
```

只要你把这条链跑顺、看懂，后面无论是扩规则、补特征，还是做错例分析，都会轻松很多。
