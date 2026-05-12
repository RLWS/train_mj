# 多任务训练方案

更新时间：`2026-05-12`

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

v1.2 将 P800 默认调度改成 `batch_interleave`，其它主要训练变量保持不变：

1. 每个 epoch 同时打开 `turn` 与 `response` train loader。
2. 按有效训练 batch 进度比例交错取 batch，而不是固定 1:1。
3. 每个 batch 仍然独立执行一次 forward/backward/optimizer step。
4. valid、checkpoint 选择、loss scale 和 class weight 逻辑保持和 v1.1 一致。

这样做的目的，是去掉 v1.1 里“每个 epoch 最后一大段总是 response 更新共享主干”的调度偏置。它不会让 response 不影响 turn；它只是让两个任务在整个 epoch 中更均匀地约束共享状态 encoder，便于判断 response 回退是否来自 epoch-sequential 覆盖效应。

注意：response train 会过滤 pass-only 样本，`DataLoader.__len__()` 仍按原始 response 行数估算，因此 P800 v1.2 显式配置 `task.interleave_batches`。当前 20w 数据根据 v1.1 训练日志使用 `turn=10150`、`response=1420`，对应实际每 epoch 训练样本约 `8308 万` turn 和 `2312 万` filtered response。

## 4. P800 启动方式

默认配置：

```bash
configs/multitask_p800.yaml
```

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

训练输出默认写入：

```text
outputs/checkpoints/multitask_p800_v1_2_interleave/
outputs/logs/multitask_p800_v1_2_interleave/
```

eval 默认产出两份报告：

```text
outputs/reports/multitask_p800_v1_2_interleave_turn_test_eval.json
outputs/reports/multitask_p800_v1_2_interleave_response_test_eval.json
```

手动评估同一个多任务 checkpoint 时，需要显式指定任务：

```bash
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_p800_v1_2_interleave/best.pt --task turn --split test --report outputs/reports/multitask_turn_test_eval.json
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_p800_v1_2_interleave/best.pt --task response --split test --report outputs/reports/multitask_response_test_eval.json
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
