# 雀魂四麻 AI 工程落地方案

## 0. 当前实现状态（2026-04-24）

本仓库已经从“纯规划阶段”进入“规则与数据链路已落地、最小训练链路已打通”的阶段。当前实际情况如下：

- 已完成：
  - 天凤 XML 解析
  - `bronze_events` 持久化
  - 回放校验
  - 基础合法动作生成
  - `turn/response` 样本构建
  - Parquet 写出与 manifest 切分
  - 最小 `turn` baseline 训练链路
  - 规则与数据链路相关测试
- 部分完成：
  - CLI 框架已建立，当前 `preprocess`、`validate-events`、`build-dataset`、`make-manifest`、`train` 已可用
  - 数据统计、失败归因和 split warning 已有第一版，但调试视图仍比较基础
- 尚未开始：
  - `eval/`
  - `serving/`

当前已验证结果：

- `pytest` 结果为 `46 passed`
- 使用 `tests/fixtures/tenhou` 实测可完成 `10` 局 XML 预处理、`3253` 条事件写入、`10/10` 个事件分片校验通过，并生成 `1542` 条 `turn` 样本、`4626` 条 `response` 样本
- `turn` 单任务本地 smoke train 已可跑通并落盘 checkpoint 与指标

更细的进度判断、模块状态和建议的下一步任务见 `PROJECT_STATUS.md`。

## 1. 目标

本文档只回答工程落地问题，默认沿用主方案中的技术结论：

- 框架统一为 `PyTorch 2.5.1+cu118`
- 当前只做四麻
- 使用“规则引擎 + 共享编码器 + 多头候选动作排序”
- 任务拆分为 `turn_head`、`response_head`、`value_head`

目标是尽快搭出一套可迭代骨架，满足：

- 本地 4070 可跑通 parser、样本构建和小模型训练
- 训练机可直接切换到正式训练
- 合法性校验贯穿数据、训练和推理
- 命令入口统一，后续加模型和加数据不需要重写主流程

## 2. 建议目录结构

```text
train_mj/
  README.md
  MAHJONG_AI_PLAN.md
  MAHJONG_AI_ENGINEERING.md
  pyproject.toml
  requirements.txt
  .env.example
  configs/
    base.yaml
    data/
      tenhou_4p_red.yaml
    model/
      encoder_small.yaml
      encoder_base.yaml
    task/
      turn.yaml
      response.yaml
      value.yaml
      multitask.yaml
    train/
      local_smoke.yaml
      local_debug.yaml
      p800_single.yaml
      p800_ddp.yaml
  data/
    raw_xml/
    bronze_events/
    silver_samples/
    manifests/
    cache/
  outputs/
    logs/
    checkpoints/
    reports/
    replay_cases/
  scripts/
    preprocess.ps1
    preprocess.sh
    train_local.ps1
    train_p800.sh
  mjai/
    __init__.py
    cli.py
    utils/
      io.py
      logging.py
      seed.py
      registry.py
      device.py
      paths.py
    data/
      tenhou/
        parser.py
        meld_decoder.py
        filters.py
      schema/
        events.py
        samples.py
      builders/
        event_builder.py
        sample_builder.py
      datasets/
        parquet_dataset.py
        collate.py
    rules/
      tiles.py
      state.py
      legal_actions.py
      hand_checker.py
      shanten.py
      scoring.py
      validators.py
    features/
      state_features.py
      history_features.py
      action_features.py
    models/
      common/
        embeddings.py
        transformer.py
      encoder.py
      heads/
        turn_head.py
        response_head.py
        value_head.py
      policy_net.py
    training/
      engine.py
      losses.py
      metrics.py
      hooks.py
      optim.py
      scheduler.py
    eval/
      offline_eval.py
      replay_eval.py
      report.py
    serving/
      inference.py
      action_adapter.py
  tests/
    unit/
      test_tiles.py
      test_meld_decoder.py
      test_state.py
      test_legal_actions.py
      test_shanten.py
    integration/
      test_parse_replay.py
      test_build_samples.py
      test_train_smoke.py
    fixtures/
      tenhou/
        basic_01.xml
        riichi_01.xml
        meld_01.xml
```

## 3. 目录职责说明

### `configs/`

只放配置，不写业务逻辑。建议把配置拆成四层：

- `data/`：规则域、路径、切分、采样
- `model/`：编码器规模、head 参数
- `task/`：当前训练任务和损失组合
- `train/`：设备、batch size、workers、precision、DDP

### `mjai/data/`

负责 XML 解析、事件标准化、样本构建和数据加载。不要把合法动作判断混进 parser。

### `mjai/rules/`

负责牌局状态、手牌增减、合法动作生成、向听和胡牌校验。这里是全项目最关键的正确性模块。

### `mjai/features/`

只负责把当前局面和候选动作转换成模型输入，不应包含规则推导。

### `mjai/models/`

负责共享编码器和多头结构。建议 `policy_net.py` 统一暴露接口：

```python
forward_turn(batch)
forward_response(batch)
forward_value(batch)
```

### `mjai/training/`

负责训练引擎、损失、日志、优化器和 checkpoint 管理。这里不要写具体规则。

### `mjai/serving/`

负责把模型输出转换成雀魂可执行动作，并在最终输出前再次做合法性过滤。

## 4. 配置设计

建议统一使用 YAML，采用“主配置 + 子配置合并”的模式。

## 4.1 配置切分

主配置示例：

```yaml
defaults:
  - data: tenhou_4p_red
  - model: encoder_small
  - task: turn
  - train: local_smoke

seed: 20260423
exp_name: turn_small_smoke
output_dir: outputs/checkpoints/${exp_name}
log_dir: outputs/logs/${exp_name}
```

`configs/data/tenhou_4p_red.yaml`

```yaml
ruleset: tenhou_4p_red
tile_encoding: mahjong_34_plus_aka
input_root: data/bronze_events
train_manifest: data/manifests/train.jsonl
valid_manifest: data/manifests/valid.jsonl
test_manifest: data/manifests/test.jsonl
max_history_len: 192
candidate_limit: 64
```

`configs/model/encoder_small.yaml`

```yaml
name: hybrid_transformer_ranker
hidden_size: 256
num_layers: 4
num_heads: 8
dropout: 0.1
state_mlp_dim: 512
action_mlp_dim: 256
share_encoder: true
heads:
  turn: true
  response: true
  value: true
```

`configs/task/turn.yaml`

```yaml
task: turn
losses:
  turn_ce: 1.0
  response_ce: 0.0
  value_mse: 0.0
label_key: turn_label
legal_key: turn_legal_actions
```

`configs/task/multitask.yaml`

```yaml
task: multitask
losses:
  turn_ce: 1.0
  response_ce: 1.0
  value_mse: 0.2
sampling:
  turn_weight: 1.0
  response_weight: 1.5
  value_weight: 0.5
```

`configs/train/local_smoke.yaml`

```yaml
device: cuda
precision: fp16
batch_size: 32
num_workers: 2
epochs: 1
grad_accum_steps: 1
distributed: false
log_every_n_steps: 20
val_every_n_steps: 100
save_every_n_steps: 200
```

`configs/train/p800_ddp.yaml`

```yaml
device: xpu
precision: bf16
batch_size: 256
num_workers: 16
epochs: 3
grad_accum_steps: 2
distributed: true
backend: nccl_or_vendor_backend
find_unused_parameters: false
```

## 4.2 配置原则

- 不在代码里硬编码路径
- 不在代码里写死设备类型
- 规则域、牌表示、历史长度、候选动作上限都可配
- `turn`、`response`、`multitask` 通过 `task` 配置切换
- 设备相关差异只出现在 `train/*.yaml`

## 5. CLI 约定

建议所有功能收口到一个入口：

```bash
python -m mjai.cli <subcommand> [args]
```

建议子命令如下。

## 5.1 数据处理

```bash
python -m mjai.cli preprocess --input data/raw_xml --output data/bronze_events --config configs/data/tenhou_4p_red.yaml
python -m mjai.cli validate-events --input data/bronze_events --report outputs/reports/event_validation.json
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --config configs/data/tenhou_4p_red.yaml
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
```

说明：

- `preprocess`：XML 转事件流
- `validate-events`：回放校验和规则一致性检查
- `build-dataset`：生成训练样本
- `make-manifest`：生成 train/valid/test 清单和统计

## 5.2 训练与评估

```bash
python -m mjai.cli train --config configs/turn_small.yaml
python -m mjai.cli train --config configs/response_small.yaml
python -m mjai.cli train --config configs/policy_multitask.yaml
python -m mjai.cli eval --ckpt outputs/checkpoints/turn_small/best.pt --split valid
python -m mjai.cli replay-eval --ckpt outputs/checkpoints/turn_small/best.pt --games data/manifests/replay_cases.txt
```

说明：

- `train`：统一训练入口，依赖配置决定当前 task
- `eval`：离线指标评估
- `replay-eval`：固定牌谱回放

## 5.3 推理与调试

```bash
python -m mjai.cli inspect-sample --dataset data/silver_samples --index 100
python -m mjai.cli infer-on-state --ckpt outputs/checkpoints/turn_small/best.pt --state-json tests/fixtures/state_case_01.json
python -m mjai.cli dump-case --game-id <id> --hand-id <id> --step-id <id>
```

说明：

- `inspect-sample`：抽查样本内容
- `infer-on-state`：手工给定状态跑一次推理
- `dump-case`：定位错误样本和错误牌谱

## 5.4 CLI 约束

- 子命令名尽量用短横线风格
- 所有命令支持 `--config`
- 所有命令支持 `--output-dir`
- 日志必须打印 `exp_name`、`git_commit`、`data_manifest`
- 所有训练命令必须落 `config.resolved.yaml`

## 6. 模型工程约定

建议主模型类定义为：

```python
class MahjongPolicyNet(nn.Module):
    def forward_turn(self, batch): ...
    def forward_response(self, batch): ...
    def forward_value(self, batch): ...
```

输入 batch 建议统一字段：

```python
{
    "state_feats": ...,
    "history_tokens": ...,
    "candidate_feats": ...,
    "candidate_mask": ...,
    "label": ...,
    "meta": ...
}
```

说明：

- `candidate_mask` 用于屏蔽非法动作
- `label` 必须是候选动作索引，不直接是牌字符串
- `meta` 保存 `game_id`、`hand_id`、`step_id`、`actor`

## 7. 数据格式约定

建议 `bronze_events` 每条记录包含：

```text
game_id
hand_id
step_id
event_type
actor
tile
meld
scores
honba
kyotaku
round_wind
seat_wind
raw_payload
```

建议 `silver_samples` 每条记录包含：

```text
sample_id
game_id
hand_id
step_id
decision_type        # turn / response / value
state_blob
history_blob
legal_actions_blob
label_action_idx
label_action_text
player_weight
ruleset_id
```

关键要求：

- `label_action_idx` 必须对应 `legal_actions_blob`
- `label_action_text` 只用于 debug，不作为训练主键
- 样本切分必须按 `game_id`

## 8. 首批开发任务清单

建议按下面顺序做，不要并行铺太大。

注：截至 `2026-04-24`，`Task 1` 到 `Task 5` 及 `preprocess` / `validate-events` 已完成第一版；`Task 6` 及以后尚未开始。详细状态见 `PROJECT_STATUS.md`。

### Task 1：项目骨架初始化

产出：

- `pyproject.toml`
- `requirements.txt`
- `mjai/` 包结构
- `configs/` 初始目录
- `tests/` 初始目录

验收标准：

- `python -m mjai.cli --help` 可运行
- `pytest` 可发现测试

### Task 2：牌表示与基础规则模块

产出：

- `tiles.py`
- `state.py`
- `legal_actions.py` 骨架
- 赤牌映射工具

验收标准：

- 能表示 34 类牌 + 赤牌
- 能完成摸牌、弃牌、碰后减张等基础状态变更
- 单测覆盖最小闭环

### Task 3：天凤副露解析与事件 schema

产出：

- `meld_decoder.py`
- `events.py`
- `parser.py` 初版

验收标准：

- 能解析 `INIT`、摸牌、弃牌、`N`、`REACH`、`DORA`、`AGARI`、`RYUUKYOKU`
- 固定 fixture 可回放

### Task 4：状态回放与合法动作生成

产出：

- `validators.py`
- `legal_actions.py` 首版
- `test_parse_replay.py`

验收标准：

- 弃牌合法性可校验
- 吃碰杠基础规则可校验
- 标签动作不在合法集合内时直接报错

### Task 5：样本构建

产出：

- `sample_builder.py`
- `samples.py`
- `parquet_dataset.py`

验收标准：

- 可从事件流构建 `turn` 和 `response` 样本
- 输出 Parquet shard
- 可按 `game_id` 生成 manifest

### Task 6：baseline 模型与训练引擎

产出：

- `encoder.py`
- `turn_head.py`
- `response_head.py`
- `policy_net.py`
- `engine.py`

验收标准：

- 本地 100 局样本能跑通 1 个 epoch
- loss 可下降
- 无 NaN

### Task 7：TensorBoard 与离线评估

产出：

- `metrics.py`
- `offline_eval.py`
- `report.py`

验收标准：

- 能输出 top-k accuracy
- 能输出动作级 precision/recall
- 能导出一个 JSON 或 Markdown 报告

### Task 8：多任务训练与共享编码器

产出：

- `multitask.yaml`
- 多头 loss 聚合逻辑
- 编码器共享与 head 路由逻辑

验收标准：

- 同一模型可切 `turn`、`response`、`multitask`
- checkpoint 可单独评估各 head

### Task 9：训练机适配

产出：

- `device.py`
- `p800_single.yaml`
- `p800_ddp.yaml`
- `train_p800.sh`

验收标准：

- 单卡可跑
- 多卡初始化可跑
- DataLoader、混合精度和 checkpoint 保存正常

## 9. 第一阶段代码冻结范围

为了避免首版过度扩张，第一阶段明确不做：

- 三麻
- 雀魂 UI 接管
- 在线对局代理
- 自博弈训练
- 完整高阶计分优化
- 大而全的数据平台

第一阶段只做：

- 可验证 parser
- 可测试 rules
- 可训练 turn/response baseline
- 可视化与离线评估

## 10. 建议的最小可运行里程碑

如果要尽快拿到第一个真正可运行版本，建议只盯下面四个里程碑。

### M1

- 解析 100 局 XML
- 回放通过
- 能生成合法动作

### M2

- 构建 1000 局 `turn` 样本
- 本地训练 `turn_head`
- 输出 TensorBoard

### M3

- 构建 `response` 样本
- 跑通 `response_head`
- 解决 `pass` 不平衡

### M4

- 接入共享编码器
- 多任务微调
- 在训练机上跑通正式实验

## 11. 当前建议

如果现在就开始开工，我建议你把开发顺序定成：

1. `rules/` 和 `data/tenhou/`
2. `sample_builder`
3. `turn_head` baseline
4. `response_head` baseline
5. 共享编码器多头化
6. P800 训练适配

这个顺序的核心原因是：麻将项目最容易失败的不是模型层，而是 parser 和 legal actions。规则层不稳，后面的训练指标没有意义。
