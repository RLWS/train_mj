# 多任务训练方案

更新时间：`2026-05-21`

## 1. 目标

v1.1 的多任务训练目标不是把 v1.0.0 的两个 checkpoint 简单拼接成一个推理文件，而是在同一次训练中共享一段状态表征，让 `turn` 与 `response` 两个候选排序任务共同更新共享参数，再分别保留任务专属 action encoder 与 head。

这一版先做可落地训练的 baseline，避免同时引入 DDP、value head、复杂 batch 混采和新特征。验收标准是：同一个 checkpoint 可以分别跑 `turn` 与 `response` 离线评估，并能和 v1.0.0 单任务 baseline 做 test 指标对照。

## 2. 模型结构

当前落地结构为：

```text
turn state feats     -> turn state projection     \
                                                   -> shared state encoder -> turn head
response state feats -> response state projection /

turn action feats     -> turn action encoder     -> turn head
response action feats -> response action encoder -> response head
```

保留任务专属投影的原因是 `turn` 和 `response` 的输入维度不同：

- `turn` state/action dim：`TURN_STATE_DIM` / `TURN_ACTION_DIM`
- `response` state/action dim：`RESPONSE_STATE_DIM` / `RESPONSE_ACTION_DIM`

因此 v1.1 的“多任务”含义是共享归一化后的状态表征主干，而不是强行要求两个任务共用同一个原始输入层或同一个 action encoder。action encoder 和候选排序 head 仍然按任务拆开，避免 response 的吃碰荣候选语义干扰 turn 的弃牌/自摸/暗杠候选语义。

实现入口：

- `mjai/models/multitask_policy_net.py`
- `task.task: multitask`
- `configs/multitask_p800.yaml`

## 3. 训练调度

v1.1 baseline 使用 epoch 级顺序调度：

1. 每个 epoch 先跑完整 `turn` train loader。
2. 再跑完整 `response` train loader。
3. 如果配置了 valid manifest，则分别跑 `turn` valid 与 `response` valid。

这不是最终的混合采样方案，但它有三个好处：

- 复用已验证的单任务 `_run_epoch` 指标、class weight、response diagnostics 和 checkpoint 逻辑。
- 避免第一版多任务同时引入 loader interleave、loss 归一化和采样比例等额外变量。
- 训练结果可以直接和 v1.0.0 单任务 run 对比，先判断共享状态 encoder 是否有收益或负迁移。

checkpoint 选择：

- `best_turn.pt`：按 `turn` valid accuracy。
- `best_response.pt`：按 response selection score，即 `non_pass_f1 - false_call_rate`。
- `best_multitask.pt` / `best.pt`：按 `turn` accuracy 与 response selection score 的简单平均。
- `final.pt`：最后一个 epoch。

v1.2 这轮实验使用 `batch_interleave`，其它主要训练变量保持不变：

1. 每个 epoch 同时打开 `turn` 与 `response` train loader。
2. 按有效训练 batch 进度比例交错取 batch，而不是固定 1:1。
3. 每个 batch 仍然独立执行一次 forward/backward/optimizer step。
4. valid、checkpoint 选择、loss scale 和 class weight 逻辑保持和 v1.1 一致。

这样做的目的，是去掉 v1.1 里“每个 epoch 最后一大段总是 response 更新共享主干”的调度偏置。它不会让 response 不影响 turn；它只是让两个任务在整个 epoch 中更均匀地约束共享状态 encoder，便于判断 response 回退是否来自 epoch-sequential 覆盖效应。

注意：response train 会过滤 pass-only 样本，`DataLoader.__len__()` 仍按原始 response 行数估算，因此 P800 v1.2 显式配置 `task.interleave_batches`。当前 20w 数据根据 v1.1 训练日志使用 `turn=10150`、`response=1420`，对应实际每 epoch 训练样本约 `8308 万` turn 和 `2312 万` filtered response。

## 4. P800 启动方式

默认配置当前回到 v1.1 epoch-sequential：

```bash
configs/multitask_p800.yaml
```

v1.2 interleave 只作为显式实验配置保留，不再放在主配置入口目录：

```bash
configs/experiments/multitask_p800_v1_2_interleave.yaml
```

该 v1.2 复现实验配置现在默认启用 DDP：

```yaml
train:
  distributed: true
  backend: vendor
  find_unused_parameters: true
```

`find_unused_parameters` 必须打开，因为 batch-interleave 每个 batch 只会经过
`turn` 或 `response` 的一条分支。手动启动时使用：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 -m mjai.cli train --config configs/experiments/multitask_p800_v1_2_interleave.yaml
```

如果通过 `start.sh` 或 docker compose 复现，配置文件中检测到
`distributed: true` 后会自动走 `torch.distributed.run`；多卡数量可用
`DDP_NPROC_PER_NODE` 覆盖。训练数据 streaming loader 会按 rank 切分 shard，
valid/eval 和 checkpoint 仍只在 rank 0 上执行。

在 P800/Linux 训练机上，从仓库根目录直接运行：

```bash
docker-compose -f docker-compose.p800.yml -f docker-compose.p800.multitask.yml up --force-recreate
```

`docker-compose.p800.multitask.yml` 会显式设置 `TASKS=multitask`、`RUN_TRAIN=1`、`RUN_EVAL=1`，并跳过 preprocess / validate / build / manifest，适合复用已有 `data/bronze_events`、`data/silver_samples` 和 `data/manifests`。

如果要临时不用 override 文件，也可以在 shell 环境变量能被 `docker-compose` 正确读取的机器上运行：

```bash
env TASKS=multitask RUN_TRAIN=1 RUN_EVAL=1 RUN_PREPROCESS=0 RUN_VALIDATE=0 RUN_BUILD_DATASET=0 RUN_MANIFEST=0 docker-compose -f docker-compose.p800.yml up --force-recreate
```

启动前可以先检查最终 compose 配置，确认环境变量已经进入容器：

```bash
docker-compose -f docker-compose.p800.yml -f docker-compose.p800.multitask.yml config | grep -E 'TASKS|RUN_TRAIN|RUN_EVAL|RUN_PREPROCESS|RUN_BUILD_DATASET|RUN_MANIFEST'
```

默认训练输出写入：

```text
outputs/checkpoints/multitask_p800_v1_1/
outputs/logs/multitask_p800_v1_1/
```

默认 eval 产出两份报告：

```text
outputs/reports/multitask_p800_v1_1_turn_test_eval.json
outputs/reports/multitask_p800_v1_1_response_test_eval.json
```

手动评估同一个多任务 checkpoint 时，需要显式指定任务：

```bash
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_p800_v1_1/best.pt --task turn --split test --report outputs/reports/multitask_turn_test_eval.json
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_p800_v1_1/best.pt --task response --split test --report outputs/reports/multitask_response_test_eval.json
```

如果需要复现 v1.2 interleave，显式覆盖多任务配置和报告路径：

```bash
env TASKS=multitask RUN_TRAIN=1 RUN_EVAL=1 RUN_PREPROCESS=0 RUN_VALIDATE=0 RUN_BUILD_DATASET=0 RUN_MANIFEST=0 \
  MULTITASK_TRAIN_CONFIG=configs/experiments/multitask_p800_v1_2_interleave.yaml \
  MULTITASK_EVAL_CKPT=outputs/checkpoints/multitask_p800_v1_2_interleave/best.pt \
  MULTITASK_TURN_EVAL_REPORT=outputs/reports/multitask_p800_v1_2_interleave_turn_test_eval.json \
  MULTITASK_RESPONSE_EVAL_REPORT=outputs/reports/multitask_p800_v1_2_interleave_response_test_eval.json \
  docker-compose -f docker-compose.p800.yml up --force-recreate
```

## 5. 首轮对照指标

多任务 v1.1 首轮训练完成后，至少对照 v1.0.0 improved test 指标：

- `turn top1/top3/top5`
- `turn discard top1`
- `turn riichi_discard top1`
- `turn ankan/kakan/tsumo` 少数类 top1
- `response multi_candidate top1`
- `response multi_candidate macro top1`
- `response non_pass_f1`
- `response false_call_rate`
- `response missed_call_rate`
- `response chi/pon top1`

判断标准：

- 如果 `turn` 小幅持平或提升，且 `response` 的 `non_pass_f1` 与 false/missed call 没有明显回退，多任务共享 encoder 可以进入 v1.1 后续优化。
- 如果 `response` 明显回退，优先怀疑共享状态主干对 response 产生负迁移；下一步尝试降低 response loss scale、减小共享层深度，或改成只共享更浅层状态投影。
- 如果 `turn` 回退而 response 持平，优先检查 epoch 顺序调度是否让 response 后训练覆盖共享参数；下一步再做 batch-level interleave 或 epoch 内交替。

## 6. 暂不纳入 v1.1 baseline 的内容

以下内容先延后，避免首轮实验变量过多：

- value head / value loss
- DDP 多卡训练
- 自动 class weight 估计
- WeightedRandomSampler 或分层采样
- 新的 state/action feature
- response calibration 默认阈值迁移

response calibration 仍然要在多任务 checkpoint 训练完成后单独跑，不应直接沿用 v1.0.0 improved 的 `pass_bias=0.3`。

## 7. 2026-05-12 P800 首轮结果

本轮 run 已归档到 `runs/2026-05-12_p800_multitask_v1_1/`，继续使用数据集 `runs/_datasets/tenhou_20w_2026-05-06/`。训练产物来自：

```text
outputs/checkpoints/multitask_p800_v1_1/
outputs/logs/multitask_p800_v1_1/
outputs/reports/multitask_p800_v1_1_*_test_eval.json
```

归档后对应路径为：

```text
runs/2026-05-12_p800_multitask_v1_1/multitask/checkpoints/
runs/2026-05-12_p800_multitask_v1_1/multitask/logs/
runs/2026-05-12_p800_multitask_v1_1/multitask/eval/
```

### 7.1 Turn 对比

`turn` 是本轮最明确的收益点。相比 2026-05-10 improved 单任务 run，multitask test 指标整体上升：

| 指标 | 2026-05-06 baseline | 2026-05-10 improved | multitask v1.1 | 相对 improved |
| --- | ---: | ---: | ---: | ---: |
| loss | `0.8206` | `0.8000` | `0.7724` | `-0.0276` |
| top1 | `70.08%` | `70.75%` | `71.69%` | `+0.94pp` |
| top3 | `94.46%` | `94.93%` | `95.38%` | `+0.45pp` |
| top5 | `98.53%` | `98.68%` | `98.84%` | `+0.16pp` |
| macro top1 | `91.34%` | `90.57%` | `91.46%` | `+0.89pp` |
| discard top1 | `69.58%` | `70.24%` | `71.18%` | `+0.94pp` |
| riichi_discard top1 | `85.34%` | `87.13%` | `88.03%` | `+0.90pp` |
| ankan top1 | `96.02%` | `89.88%` | `91.86%` | `+1.98pp` |
| kakan top1 | `97.26%` | `96.27%` | `97.71%` | `+1.45pp` |
| non_discard top1 | `90.49%` | `91.37%` | `92.07%` | `+0.71pp` |

结论：multitask 的共享状态主干对 `turn` 是正收益，收益不只来自总体 top1，`discard`、`riichi_discard`、`top3/top5` 和 `non_discard` 都同步改善。`ankan` 相比 2026-05-10 improved 有恢复，但仍低于 2026-05-06 baseline，因此少数类排查还不能完全关闭。

### 7.2 Response raw eval 对比

`response` 的 raw eval 不是全面提升。下面同时列出 multitask 的 `best.pt` 和 `best_response.pt`；其中 `best.pt` 是 turn 与 response 的折中 checkpoint，`best_response.pt` 是按 response selection score 选出的 checkpoint。

| 指标 | 2026-05-10 improved | multitask `best.pt` | multitask `best_response.pt` | best_response vs improved |
| --- | ---: | ---: | ---: | ---: |
| overall top1 | `98.770%` | `98.681%` | `98.732%` | `-0.037pp` |
| macro top1 | `90.60%` | `86.60%` | `87.42%` | `-3.18pp` |
| multi-candidate top1 | `86.86%` | `85.91%` | `86.46%` | `-0.40pp` |
| multi-candidate macro top1 | `88.43%` | `84.20%` | `85.18%` | `-3.25pp` |
| non-pass precision | `59.04%` | `57.04%` | `58.27%` | `-0.77pp` |
| non-pass recall | `92.72%` | `93.94%` | `92.45%` | `-0.27pp` |
| non-pass F1 | `72.14%` | `70.98%` | `71.49%` | `-0.66pp` |
| false call rate | `1.105%` | `1.216%` | `1.137%` | `+0.032pp` |
| missed call rate | `7.28%` | `6.06%` | `7.55%` | `+0.27pp` |
| false calls | `335,932` | `369,600` | `345,760` | `+9,828` |
| missed calls | `38,033` | `31,630` | `39,434` | `+1,401` |
| chi top1 | `81.98%` | `85.21%` | `81.76%` | `-0.22pp` |
| pon top1 | `94.75%` | `95.12%` | `94.25%` | `-0.50pp` |
| daiminkan top1 | `68.17%` | `40.64%` | `49.77%` | `-18.41pp` |

结论：`best_response.pt` 比 multitask `best.pt` 更接近 improved response，false call 和 overall top1 都有恢复，但仍没有超过 2026-05-10 improved。它在 raw test 上仍有更多 false call、更多 missed call、更低 `non_pass_f1`，并且 `multi-candidate macro top1` 与 `daiminkan` 明显偏弱。因此当前不能用 multitask response 直接替代 2026-05-10 improved response。

需要注意，response 对比不是纯粹的“是否多任务”变量：2026-05-10 improved response 单任务配置是 `hidden_size=256`，而本轮 multitask 使用 `hidden_size=512` 的共享主干。即便容量更大，raw response 仍没有超过 improved，因此下一步应优先做校准和调度实验，而不是直接扩大模型。

### 7.3 Response calibration 对比

本轮已经补跑 valid/test calibration：

- `multitask_best_response_calibration_valid.json`：校准 `best_response.pt`。
- `multitask_best_response_calibration_test.json`：在 test 上复核 `best_response.pt`。
- `multitask_best_calibration_valid.json`：校准 `best.pt`，用于诊断折中 checkpoint。
- `multitask_best_calibration_test.json`：在 test 上诊断 `best.pt`。

valid 关键 bias 点如下：

| checkpoint | pass_bias | false call | missed call | non-pass F1 | multi top1 | chi / pon / daiminkan top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | `0.0` | `1.103%` | `7.35%` | `72.14%` | `86.86%` | `81.7 / 94.7 / 66.7` |
| multitask best | `0.0` | `1.215%` | `6.11%` | `70.97%` | `85.91%` | `85.0 / 95.1 / 42.0` |
| multitask best_response | `0.0` | `1.136%` | `7.61%` | `71.48%` | `86.45%` | `81.4 / 94.3 / 51.2` |
| improved | `0.2` | `0.982%` | `8.67%` | `73.52%` | `87.91%` | `79.5 / 93.6 / 62.5` |
| multitask best_response | `0.2` | `0.997%` | `9.16%` | `73.01%` | `87.65%` | `78.8 / 93.0 / 42.7` |
| improved | `0.6` | `0.760%` | `11.83%` | `75.88%` | `89.69%` | `74.2 / 90.9 / 50.3` |
| multitask best_response | `0.6` | `0.749%` | `12.86%` | `75.54%` | `89.61%` | `73.0 / 89.6 / 28.4` |
| improved | `1.0` | `0.575%` | `15.75%` | `77.38%` | `90.94%` | `67.9 / 87.3 / 37.4` |
| multitask best_response | `1.0` | `0.544%` | `17.55%` | `77.02%` | `90.94%` | `65.9 / 84.9 / 18.1` |

按“第一个进入目标 false call 档位”的口径：

| checkpoint | target false call | first bias | false call | missed call | non-pass F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| improved | `1.0%` | `0.2` | `0.982%` | `8.67%` | `73.52%` |
| multitask best_response | `1.0%` | `0.2` | `0.997%` | `9.16%` | `73.01%` |
| improved | `0.8%` | `0.6` | `0.760%` | `11.83%` | `75.88%` |
| multitask best_response | `0.8%` | `0.6` | `0.749%` | `12.86%` | `75.54%` |

test 复核后的可比点如下：

| checkpoint | pass_bias | false call | missed call | non-pass F1 | multi top1 | chi / pon / daiminkan top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | `0.3` | `0.925%` | `9.29%` | `74.18%` | `88.40%` | `78.4 / 93.1 / 61.6` |
| multitask best_response | `0.3` | `0.933%` | `9.91%` | `73.73%` | `88.20%` | `77.8 / 92.2 / 40.3` |
| improved | `0.4` | `0.869%` | `10.06%` | `74.80%` | `88.85%` | `77.1 / 92.4 / 58.7` |
| multitask best_response | `0.4` | `0.869%` | `10.83%` | `74.38%` | `88.70%` | `76.3 / 91.4 / 36.3` |
| improved | `0.6` | `0.763%` | `11.76%` | `75.86%` | `89.67%` | `74.4 / 90.9 / 50.9` |
| multitask best_response | `0.6` | `0.750%` | `12.83%` | `75.54%` | `89.61%` | `73.1 / 89.5 / 27.9` |
| improved | `1.0` | `0.578%` | `15.65%` | `77.39%` | `90.93%` | `68.1 / 87.3 / 38.1` |
| multitask best_response | `1.0` | `0.543%` | `17.51%` | `77.06%` | `90.96%` | `66.1 / 84.9 / 18.2` |

`best.pt` 的 test calibration 也确认了它更偏召回，而不是更好的 response 默认点：`pass_bias=0.6` 时 false call 为 `0.853%`、missed call 为 `9.95%`、`non_pass_f1=75.13%`，但 `daiminkan top1` 只有 `22.2%`；`pass_bias=1.0` 时 `non_pass_f1` 升到 `77.11%`，missed call 也升到 `13.56%`，`daiminkan top1` 降到 `11.1%`。

结论：valid 和 test 曲线一致。校准后 `best_response.pt` 可以接近 improved response，但在相同或相近 false-call 区间仍略弱：`non_pass_f1` 更低、missed call 更高，`chi/pon` 略低，`daiminkan` 明显低。自动 target selection 会倾向选择更大的 `pass_bias` 来抬高 `non_pass_f1`，但这会把 missed call 和少数类质量压坏；不能只按 report 里的 selected point 决定默认 operating point。`best.pt` 适合诊断 epoch-sequential multitask 的偏召回行为，不适合作为 response 默认候选。

### 7.4 当前决策

- `turn`：multitask v1.1 可以作为新的候选 baseline，后续优先用 `runs/2026-05-12_p800_multitask_v1_1/multitask/checkpoints/best.pt` 或 `best_turn.pt` 做进一步验证。
- `response`：继续保留 `runs/2026-05-10_p800_improved/response/checkpoints/best.pt` 作为当前默认候选。multitask `best_response.pt` 的 raw test、valid calibration 和 test calibration 均未证明它能替代 improved。
- 校准：multitask `best_response.pt` 的实验候选点可记录为 `pass_bias=0.2`（test `false_call=0.998%`、`missed=9.07%`、`F1=73.02%`）和 `pass_bias=0.6`（test `false_call=0.750%`、`missed=12.83%`、`F1=75.54%`）。它们可用于后续对照，但不建议直接沿用为默认；v1.0.0 improved 的默认 `pass_bias=0.3` 仍是当前 serving 候选。
- 训练策略：test calibration 没有反转 response 结论。下一轮优先尝试调整 epoch-sequential 调度、降低 `response_ce` loss scale，或单独处理 `daiminkan`/少数类质量；不要只靠更大的 `pass_bias` 抬高 `non_pass_f1`。

### 7.5 Multitask response calibration 使用说明

多任务 checkpoint 下，`calibrate-response` 的命令入口和单任务 response 基本一致，不需要额外传 `--task response`。当前实现会读取 checkpoint 内的 `task.task`：

- 如果是 `task=response`，按单任务 `MahjongPolicyNet` 加载。
- 如果是 `task=multitask`，按 `MultitaskMahjongPolicyNet` 加载，并固定调用 `forward_response`。
- 数据集、candidate limit 和 batch size 都走 response 分支：`StreamingManifestResponseDataset`、`response_candidate_limit`、`response_batch_size`。

因此多任务场景真正需要注意的是 checkpoint 选择，而不是命令参数。用于 response calibration 时，优先校准 `best_response.pt`，不要默认拿 `best.pt`。`best.pt` / `best_multitask.pt` 是 turn accuracy 与 response score 的折中选择，可能不是 response operating point 的最佳基准。

建议流程：

1. 先跑 `best_response.pt` 的 raw test eval，确认它和 `best.pt` 的 response 行为是否不同。
2. 在 valid split 上扫描 `best_response.pt` 的 pass bias。
3. 根据 valid 选择候选 bias 后，在 test split 上复核同一组 bias。
4. 和 2026-05-10 improved response 在相同 false-call 目标下比较，不直接迁移旧默认 `pass_bias=0.3`。

本轮复现命令如下：

```powershell
python -m mjai.cli calibrate-response `
  --ckpt runs/2026-05-12_p800_multitask_v1_1/multitask/checkpoints/best_response.pt `
  --split valid `
  --manifest runs/_datasets/tenhou_20w_2026-05-06/valid.jsonl `
  --report runs/2026-05-12_p800_multitask_v1_1/multitask/calibration/multitask_best_response_calibration_valid.json `
  --pass-biases 0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.30,0.40,0.60,0.80,1.00 `
  --false-call-targets 1%,0.8%,0.5%
```

test 复核已经按同一组 bias 扫描完成：

```powershell
python -m mjai.cli calibrate-response `
  --ckpt runs/2026-05-12_p800_multitask_v1_1/multitask/checkpoints/best_response.pt `
  --split test `
  --manifest runs/_datasets/tenhou_20w_2026-05-06/test.jsonl `
  --report runs/2026-05-12_p800_multitask_v1_1/multitask/calibration/multitask_best_response_calibration_test.json `
  --pass-biases 0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.30,0.40,0.60,0.80,1.00 `
  --false-call-targets 1%,0.8%,0.5%
```

如果后续新 run 的 valid 选择出其他 bias，例如 `0.12` 或 `0.20`，把它们追加到 test 的 `--pass-biases`。最终比较时不要只看 `non_pass_f1`，还要同时看 `false_call_rate`、`missed_call_rate`、`multi_candidate_top1_accuracy` 和 `per_kind_top1_accuracy` 里的 `chi/pon/daiminkan`。

## 8. 2026-05-13 P800 v1.2 interleave 结果

本轮 run 已归档到 `runs/2026-05-13_p800_multitask_v1_2_interleave/`，继续使用数据集 `runs/_datasets/tenhou_20w_2026-05-06/`。训练产物来自：

```text
outputs/checkpoints/multitask_p800_v1_2_interleave/
outputs/logs/multitask_p800_v1_2_interleave/
outputs/reports/multitask_p800_v1_2_interleave_*_test_eval.json
```

归档后对应路径为：

```text
runs/2026-05-13_p800_multitask_v1_2_interleave/multitask/checkpoints/
runs/2026-05-13_p800_multitask_v1_2_interleave/multitask/logs/
runs/2026-05-13_p800_multitask_v1_2_interleave/multitask/eval/
```

本轮使用 `batch_interleave`，显式按有效训练 batch 配置 `turn=10150`、`response=1420`。训练样本计数仍是 `turn_train_samples=83,077,288` 与 filtered `response_train_samples=23,120,538`，因此这轮主要验证调度方式，而不是数据或特征变化。

### 8.1 Turn raw test 对比

| 指标 | v1.0.0 improved | v1.1 multitask | v1.2 interleave | v1.2 vs v1.0.0 | v1.2 vs v1.1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| loss | `0.8000` | `0.7724` | `0.7720` | `-0.0280` | `-0.0004` |
| top1 | `70.75%` | `71.69%` | `71.80%` | `+1.05pp` | `+0.12pp` |
| top3 | `94.93%` | `95.38%` | `95.39%` | `+0.46pp` | `+0.01pp` |
| top5 | `98.68%` | `98.84%` | `98.85%` | `+0.17pp` | `+0.01pp` |
| macro top1 | `90.57%` | `91.46%` | `90.78%` | `+0.21pp` | `-0.68pp` |
| discard top1 | `70.24%` | `71.18%` | `71.36%` | `+1.12pp` | `+0.18pp` |
| riichi_discard top1 | `87.13%` | `88.03%` | `84.13%` | `-2.99pp` | `-3.89pp` |
| ankan top1 | `89.88%` | `91.86%` | `92.10%` | `+2.22pp` | `+0.24pp` |
| kakan top1 | `96.27%` | `97.71%` | `97.13%` | `+0.87pp` | `-0.58pp` |
| non_discard top1 | `91.37%` | `92.07%` | `89.61%` | `-1.76pp` | `-2.47pp` |

补跑 `best_turn.pt` 后，v1.2 的少数类回退大部分恢复：

| 指标 | v1.1 `best.pt` | v1.2 `best.pt` | v1.2 `best_turn.pt` |
| --- | ---: | ---: | ---: |
| top1 | `71.69%` | `71.80%` | `71.73%` |
| macro top1 | `91.46%` | `90.78%` | `91.25%` |
| discard top1 | `71.18%` | `71.36%` | `71.25%` |
| riichi_discard top1 | `88.03%` | `84.13%` | `87.01%` |
| ankan top1 | `91.86%` | `92.10%` | `92.31%` |
| kakan top1 | `97.71%` | `97.13%` | `96.93%` |
| non_discard top1 | `92.07%` | `89.61%` | `91.43%` |

结论：v1.2 `best_turn.pt` 是合理的 turn 候选，但仍不是对 v1.1 的明确替代。它保留了略高的 overall top1，并进一步提升 `ankan`，但 v1.1 仍有更好的 macro、`riichi_discard`、`kakan` 和 `non_discard`。如果目标是稳健 turn 行为，v1.1 仍更均衡；如果目标是 overall top1 与 discard/ankan，v1.2 `best_turn.pt` 可以进入下一轮候选。

### 8.2 Response raw test 对比

| 指标 | v1.0.0 improved | v1.1 `best.pt` | v1.1 `best_response.pt` | v1.2 `best.pt` | v1.2 `best_response.pt` |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall top1 | `98.770%` | `98.681%` | `98.732%` | `98.693%` | `98.693%` |
| macro top1 | `90.60%` | `86.60%` | `87.42%` | `90.17%` | `90.17%` |
| multi-candidate top1 | `86.86%` | `85.91%` | `86.46%` | `86.04%` | `86.04%` |
| multi-candidate macro top1 | `88.43%` | `84.20%` | `85.18%` | `87.80%` | `87.80%` |
| non-pass precision | `59.04%` | `57.04%` | `58.27%` | `57.28%` | `57.28%` |
| non-pass recall | `92.72%` | `93.94%` | `92.45%` | `93.93%` | `93.93%` |
| non-pass F1 | `72.14%` | `70.98%` | `71.49%` | `71.17%` | `71.17%` |
| false call rate | `1.105%` | `1.216%` | `1.137%` | `1.203%` | `1.203%` |
| missed call rate | `7.28%` | `6.06%` | `7.55%` | `6.07%` | `6.07%` |
| false / missed calls | `335,932 / 38,033` | `369,600 / 31,630` | `345,760 / 39,434` | `365,833 / 31,684` | `365,833 / 31,684` |
| chi / pon top1 | `81.98% / 94.75%` | `85.21% / 95.12%` | `81.76% / 94.25%` | `84.84% / 95.35%` | `84.84% / 95.35%` |
| daiminkan top1 | `68.17%` | `40.64%` | `49.77%` | `62.32%` | `62.32%` |

结论：`batch_interleave` 明确修复了 v1.1 `best.pt` 的一部分 response 问题，尤其是 `macro top1`、`multi-candidate macro top1` 和 `daiminkan`。但相对 v1.0.0 improved，v1.2 `best_response.pt` 仍然有更低的 multi-candidate top1、更低的 `non_pass_f1`、更高 false call，以及仍偏弱的 `daiminkan`。因此 response 默认模型继续保留 v1.0.0 improved。

需要特别注意：v1.2 的 `best_response.pt` raw test 与 `best.pt` 完全一致。这说明本轮 response selection 没有像 v1.1 那样选出行为明显不同的 response checkpoint；v1.2 的 response 结论可以直接按上述 `best.pt`/`best_response.pt` 结果判断。

### 8.3 Response calibration 对比

v1.2 `best_response.pt` 的 valid/test 曲线一致。test 关键 bias 点如下：

| checkpoint | pass_bias | false call | missed call | non-pass F1 | multi top1 | chi / pon / daiminkan top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | `0.3` | `0.925%` | `9.29%` | `74.18%` | `88.40%` | `78.4 / 93.1 / 61.6` |
| v1.2 best_response | `0.3` | `1.026%` | `7.74%` | `73.23%` | `87.61%` | `81.9 / 94.0 / 51.1` |
| improved | `0.4` | `0.869%` | `10.06%` | `74.80%` | `88.85%` | `77.1 / 92.4 / 58.7` |
| v1.2 best_response | `0.4` | `0.969%` | `8.39%` | `73.87%` | `88.09%` | `80.7 / 93.5 / 47.2` |
| improved | `0.6` | `0.763%` | `11.76%` | `75.86%` | `89.67%` | `74.4 / 90.9 / 50.9` |
| v1.2 best_response | `0.6` | `0.861%` | `9.80%` | `75.07%` | `88.98%` | `78.4 / 92.3 / 40.2` |
| improved | `0.8` | `0.665%` | `13.62%` | `76.74%` | `90.36%` | `71.3 / 89.2 / 44.1` |
| v1.2 best_response | `0.8` | `0.759%` | `11.45%` | `76.09%` | `89.76%` | `75.7 / 90.9 / 32.7` |

同 bias 下，v1.2 主要优势是 missed call 更低、`chi/pon` 更高；主要代价是 false call 更高、F1 和 multi top1 更低，并且 `daiminkan` 明显更弱。`pass_bias=0.4` 虽然进入 `false_call <= 1%`，但仍弱于 improved 默认 `0.3` 的 F1/multi top1，且 `daiminkan` 从 `61.6%` 降到 `47.2%`。

自动 target selection 在 v1.2 valid/test 上都会选择 `pass_bias=1.0` 作为 `1.0%` 和 `0.8%` false-call 目标下的最高 F1 点；test 为 `false_call=0.665%`、`missed=13.28%`、`non_pass_f1=76.93%`。这个点过于保守，`daiminkan top1` 只有 `25.6%`，不适合作为默认 operating point。

### 8.4 当前决策

- `turn`：v1.2 `best_turn.pt` 是合理候选，但 v1.1 仍是更均衡的 turn baseline。后续如果要选 v1.2，应明确接受 macro 与 `non_discard` 小幅下降，换取 overall top1、discard 与 ankan。
- `response`：继续保留 `runs/2026-05-10_p800_improved/response/checkpoints/best.pt` 作为默认候选。v1.2 的 response 只说明 interleave 比 v1.1 `best.pt` 更健康，不说明它已经超过单任务 improved。
- 调度：`batch_interleave` 去掉了 epoch-sequential 的顺序覆盖变量，并改善了部分 response macro 指标，但没有解决 response 的核心 false/missed/F1/少数类取舍。因此默认 P800 多任务配置已回到 v1.1 `epoch_sequential`；v1.2 保留为 `configs/experiments/multitask_p800_v1_2_interleave.yaml`，只在需要复现实验或继续验证调度变量时显式使用。

## 9. 2026-05-20 commit 7a7ad21 v1.1 复训结果

本轮 run 已归档到 `runs/2026-05-20_p800_multitask_v1_1_7a7ad21/`，继续使用数据集 `runs/_datasets/tenhou_20w_2026-05-06/`。训练配置仍是 `multitask_p800_v1_1` 的 `epoch_sequential`，但代码基准是 `7a7ad21`，因此单独用 commit 号区分归档目录。完整分析见该 run 目录下的 `analysis.md`。

已归档内容包括：

```text
runs/2026-05-20_p800_multitask_v1_1_7a7ad21/multitask/checkpoints/
runs/2026-05-20_p800_multitask_v1_1_7a7ad21/multitask/logs/
runs/2026-05-20_p800_multitask_v1_1_7a7ad21/multitask/eval/
runs/2026-05-20_p800_multitask_v1_1_7a7ad21/multitask/calibration/
```

### 9.1 Turn 结论

本轮 `best_turn.pt` 是当前 20w run 里 overall 指标最高的 turn 候选：test `top1=71.88%`、`top3=95.59%`、`top5=98.92%`、`discard top1=71.42%`。相比 old v1.2 `best_turn.pt`，overall top1 提升约 `+0.15pp`，相比 old v1.1 提升约 `+0.20pp`。

代价是少数类稳定性不足：`riichi_discard=85.94%`，低于 old v1.1 的 `88.03%` 和 old v1.2 `best_turn.pt` 的 `87.01%`；`non_discard=90.73%`，也低于 old v1.1 的 `92.07%` 和 old v1.2 的 `91.43%`。

当前判断：`7a7ad21 best_turn.pt` 是 overall/discard 取向的候选，但 old v1.1 仍是更均衡的 turn baseline。

### 9.2 Response 结论

本轮 response raw test 仍不能替代 2026-05-10 improved 单任务 checkpoint：

| 指标 | improved | 7a7ad21 `best_response.pt` |
| --- | ---: | ---: |
| multi-candidate top1 | `86.86%` | `86.41%` |
| multi-candidate macro top1 | `88.43%` | `87.60%` |
| non-pass F1 | `72.14%` | `71.41%` |
| false call rate | `1.105%` | `1.142%` |
| missed call rate | `7.28%` | `7.53%` |
| chi / pon top1 | `81.98% / 94.75%` | `81.03% / 94.85%` |
| daiminkan top1 | `68.17%` | `64.59%` |

相对 old v1.1 `best_response.pt`，本轮 `daiminkan` 从 `49.77%` 恢复到 `64.59%`，说明新代码/特征状态对 rare action 有帮助；但核心 default-serving 口径仍落后于 improved。

`best.pt` 的 response macro 和 `daiminkan` 更高，raw test `macro=91.04%`、`daiminkan=71.92%`，但它的 F1、false call 和 missed call 都弱于 `best_response.pt`，因此只适合作为诊断 checkpoint。

### 9.3 Calibration 结论

本轮 valid/test calibration 都已归档：

- `multitask_p800_v1_1_best_calibration_valid.json`
- `multitask_p800_v1_1_best_response_calibration_valid.json`
- `multitask_p800_v1_1_best_calibration_test.json`
- `multitask_p800_v1_1_best_response_calibration_test.json`

test 对比没有反转 response 结论。以 `best_response.pt` 为例：

| checkpoint | pass_bias | false call | missed call | non-pass F1 | multi top1 | daiminkan top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | `0.3` | `0.925%` | `9.29%` | `74.18%` | `88.40%` | `61.6%` |
| 7a7ad21 best_response | `0.3` | `0.956%` | `9.55%` | `73.51%` | `88.02%` | `56.7%` |
| improved | `0.4` | `0.869%` | `10.06%` | `74.80%` | `88.85%` | `58.7%` |
| 7a7ad21 best_response | `0.4` | `0.897%` | `10.32%` | `74.15%` | `88.50%` | `53.0%` |
| improved | `0.6` | `0.763%` | `11.76%` | `75.86%` | `89.67%` | `50.9%` |
| 7a7ad21 best_response | `0.6` | `0.786%` | `12.01%` | `75.29%` | `89.38%` | `43.7%` |

同 bias 下，`7a7ad21 best_response.pt` 在 false call、missed call、F1、multi top1 和 `daiminkan` 上都弱于 improved。`best.pt` 保留了更多 `daiminkan`，test `pass_bias=0.3` 时为 `66.4%`，但同时只有 `F1=73.17%`、`multi top1=87.78%`，仍不适合作为默认 response checkpoint。

当前结论不变：response 默认继续使用 2026-05-10 improved；`7a7ad21` 这轮主要作为 turn overall 候选和 response rare-action 诊断证据。
