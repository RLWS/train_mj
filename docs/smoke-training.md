# Smoke Train 与调试上手教程

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

当前流程依赖本地可用的 Python / PyTorch 环境。

## 3. 第一步：准备 smoke 数据

直接使用仓库自带的 Tenhou XML 夹具：

```powershell
python -m mjai.cli preprocess --input tests/fixtures/tenhou --output data/bronze_events
python -m mjai.cli validate-events --input data/bronze_events --report outputs/reports/smoke_event_validation.json
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 1 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
```

这里推荐先单独跑 `validate-events`，确认 `files_failed=0` 后，再在 `build-dataset` 使用 `--skip-validation`，避免同一批事件在构建样本时重复回放校验。`build-dataset` 现在会输出 `elapsed_s`、`games_per_s` 和 `samples_per_s`，可以直接用来判断构建吞吐。

当前 `silver_samples` parquet 会写入离线预编码的 `state_features_bin` 和 `candidate_features_bin`。`turn` 的 discard hand-shape 特征，包括向听数、有效牌数量等 CPU 密集计算，已经从训练期 DataLoader 移到 `build-dataset` 阶段。更新到这个 schema 后，旧的 `data/silver_samples` 需要删除或覆盖后重新生成，并重新执行 `make-manifest`。

执行完成后会得到：

- 事件分片目录：[data/bronze_events](../data/bronze_events)
- 事件校验报告：[outputs/reports/smoke_event_validation.json](../outputs/reports/smoke_event_validation.json)
- 样本分片目录：[data/silver_samples](../data/silver_samples)
- manifest 目录：[data/manifests](../data/manifests)

最关键的文件是：

- [train.jsonl](../data/manifests/train.jsonl)
- [valid.jsonl](../data/manifests/valid.jsonl)
- [test.jsonl](../data/manifests/test.jsonl)

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

因为候选特征已经离线预编码，训练阶段不再需要用很多 DataLoader worker 做特征计算。Windows 本地配置默认 `num_workers: 0`，主要避免 worker spawn 后复制整份 manifest 数据集；Linux/P800 上可以从较小的 worker 数起步，再看训练日志里的 `data_wait` 是否仍然明显。

## 4. 第二步：跑 turn smoke train

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

- checkpoint：[outputs/checkpoints/turn_local_smoke](../outputs/checkpoints/turn_local_smoke)
- log：[outputs/logs/turn_local_smoke](../outputs/logs/turn_local_smoke)

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

命令：

```powershell
python -m mjai.cli train --config configs/response_local_smoke.yaml
```

配置文件：

- [response_local_smoke.yaml](../configs/response_local_smoke.yaml)

默认输出位置：

- checkpoint：[outputs/checkpoints/response_local_smoke](../outputs/checkpoints/response_local_smoke)
- log：[outputs/logs/response_local_smoke](../outputs/logs/response_local_smoke)

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

其中 `misclassified_examples` 现在已经带有：

- 样本定位信息：`sample_id / game_id / kyoku_index / step_index / actor`
- 可直接执行的 `inspect-sample --sample-id ...`
- 可直接执行的 `dump-case --bronze-dir ... --game-id ... --kyoku-id ... --step-id ...`

也就是说，评估报告里的错例已经可以直接跳回样本和原始事件上下文。

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
```

`configs/response_local_smoke.yaml` 默认设置了 `train.response_min_train_candidate_count: 2`。response 训练会过滤 `candidate_count == 1` 的 pass-only 样本，valid/test eval 仍保留全量样本。

注意：response 候选动作特征已经包含 call 前后向听、有效牌、赤牌/dora 消耗和 chi/pon 形状等离线预编码特征。修改这些特征或更新代码后，需要重新运行 `build-dataset` 和 `make-manifest`，否则旧 `candidate_features_bin` 的维度会和当前模型不匹配。

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
