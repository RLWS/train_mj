# v1.1 多任务训练方案

更新时间：`2026-05-10`

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

## 4. P800 启动方式

默认配置：

```bash
configs/multitask_p800.yaml
```

在 P800/Linux 训练机上，从仓库根目录直接运行：

```bash
TASKS=multitask RUN_TRAIN=1 RUN_EVAL=1 docker compose -f docker-compose.p800.yml up
```

如果已经有 `data/bronze_events`、`data/silver_samples` 和 `data/manifests`，保持 compose 默认即可复用现有数据：

```bash
TASKS=multitask RUN_TRAIN=1 RUN_EVAL=1 RUN_PREPROCESS=0 RUN_VALIDATE=0 RUN_BUILD_DATASET=0 RUN_MANIFEST=0 docker compose -f docker-compose.p800.yml up
```

训练输出默认写入：

```text
outputs/checkpoints/multitask_p800_v1_1/
outputs/logs/multitask_p800_v1_1/
```

eval 默认产出两份报告：

```text
outputs/reports/multitask_p800_v1_1_turn_test_eval.json
outputs/reports/multitask_p800_v1_1_response_test_eval.json
```

手动评估同一个多任务 checkpoint 时，需要显式指定任务：

```bash
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_p800_v1_1/best.pt --task turn --split test --report outputs/reports/multitask_turn_test_eval.json
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_p800_v1_1/best.pt --task response --split test --report outputs/reports/multitask_response_test_eval.json
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
