# Smoke Train 与调试上手教程

这份教程带你完整走一遍当前仓库已经打通的最小闭环，目标不是只把命令跑通，而是让你熟悉现在项目里这几类对象之间的关系：

- `bronze_events`：事件分片
- `silver_samples`：监督学习样本
- `manifests`：训练/验证切分清单
- `turn` / `response` smoke train
- `inspect-sample`：看单条样本
- `dump-case`：回到原始事件上下文

适用目录：

- 仓库根目录 [train_mj](F:\Code\AI\train_study\train_mj)

## 1. 你会得到什么

跑完这篇之后，你应该能做到：

1. 用仓库自带夹具生成一套最小 smoke 数据。
2. 分别训练 `turn` 和 `response` 两个最小 baseline。
3. 找到训练输出和评估输入的位置。
4. 用 `inspect-sample` 抽查一条样本，读懂 `sample_id`。
5. 用 `dump-case` 从 `sample_id` 反查到原始事件上下文。

## 2. 前置条件

确保当前终端位于仓库根目录：

```powershell
cd F:\Code\AI\train_study\train_mj
```

当前流程依赖本地可用的 Python / PyTorch 环境。

## 3. 第一步：准备 smoke 数据

直接使用仓库自带的 Tenhou XML 夹具：

```powershell
python -m mjai.cli preprocess --input tests/fixtures/tenhou --output data/bronze_events
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
```

执行完成后会得到：

- 事件分片目录：[data/bronze_events](F:\Code\AI\train_study\train_mj\data\bronze_events)
- 样本分片目录：[data/silver_samples](F:\Code\AI\train_study\train_mj\data\silver_samples)
- manifest 目录：[data/manifests](F:\Code\AI\train_study\train_mj\data\manifests)

最关键的文件是：

- [train.jsonl](F:\Code\AI\train_study\train_mj\data\manifests\train.jsonl)
- [valid.jsonl](F:\Code\AI\train_study\train_mj\data\manifests\valid.jsonl)
- [test.jsonl](F:\Code\AI\train_study\train_mj\data\manifests\test.jsonl)

这一步对应的含义是：

- `preprocess`：XML -> 标准化事件
- `build-dataset`：事件 -> `turn/response` 样本
- `make-manifest`：样本分片 -> 训练/验证/测试切分

## 4. 第二步：跑 turn smoke train

命令：

```powershell
python -m mjai.cli train --config configs/turn_local_smoke.yaml
```

配置文件：

- [turn_local_smoke.yaml](F:\Code\AI\train_study\train_mj\configs\turn_local_smoke.yaml)

这个配置默认读取：

- `data/manifests/train.jsonl`
- `data/manifests/valid.jsonl`

默认输出位置：

- checkpoint：[turn_local_smoke](F:\Code\AI\train_study\train_mj\outputs\checkpoints\turn_local_smoke)
- log：[turn_local_smoke](F:\Code\AI\train_study\train_mj\outputs\logs\turn_local_smoke)

重点看这些文件：

- `config.resolved.yaml`
- `train_metrics.json`
- `best.pt`
- `final.pt`
- `epoch_metrics.jsonl`

## 5. 第三步：跑 response smoke train

命令：

```powershell
python -m mjai.cli train --config configs/response_local_smoke.yaml
```

配置文件：

- [response_local_smoke.yaml](F:\Code\AI\train_study\train_mj\configs\response_local_smoke.yaml)

默认输出位置：

- checkpoint：[response_local_smoke](F:\Code\AI\train_study\train_mj\outputs\checkpoints\response_local_smoke)
- log：[response_local_smoke](F:\Code\AI\train_study\train_mj\outputs\logs\response_local_smoke)

和 `turn` 一样，至少确认这些文件存在：

- `config.resolved.yaml`
- `train_metrics.json`
- `best.pt`
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
python -m mjai.cli eval --ckpt outputs/checkpoints/turn_local_smoke/best.pt --split valid --report outputs/checkpoints/turn_local_smoke/eval_valid.json
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

其中 `misclassified_examples` 就是后面把“错例 -> dump-case”完全串起来的基础。

## 10. 一套建议你自己手动跑的完整命令

如果你就是想完整熟悉当前项目，建议按下面顺序走一遍：

```powershell
python -m mjai.cli preprocess --input tests/fixtures/tenhou --output data/bronze_events
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests

python -m mjai.cli train --config configs/turn_local_smoke.yaml
python -m mjai.cli train --config configs/response_local_smoke.yaml

python -m mjai.cli inspect-sample --dataset data/silver_samples --index 0
python -m mjai.cli dump-case --game-id 2023010100gm-00e1-0000-8a80056a --kyoku-id 0 --step-id 2
```

如果你再多走一步，把评估也跑上：

```powershell
python -m mjai.cli eval --ckpt outputs/checkpoints/turn_local_smoke/best.pt --split valid --report outputs/checkpoints/turn_local_smoke/eval_valid.json
python -m mjai.cli eval --ckpt outputs/checkpoints/response_local_smoke/best.pt --split valid --report outputs/checkpoints/response_local_smoke/eval_valid.json
```

## 11. 你现在最值得建立的心智模型

当前仓库虽然已经有 train/eval/inspect/dump-case，但它还不是一个“完整麻将 AI 系统”，而是一个已经打通最小闭环的工程骨架。

你现在最应该熟悉的是这条链：

```text
Tenhou XML
-> bronze_events
-> silver_samples
-> manifest
-> train turn/response
-> inspect-sample
-> dump-case
```

只要你把这条链跑顺、看懂，后面无论是扩规则、补特征，还是做错例分析，都会轻松很多。
