# 雀魂四麻 AI 工程指南

> 2026-10-28 更新：当前代码使用 `full_state_v2` 特征（turn/response state 11969/12012，候选 89/91），旧 bronze/silver/manifest/checkpoint 不兼容，需从原始 XML 重建。流程见 [full-state-training.md](full-state-training.md)，最终分析见 [training-analysis-2026-10-28.md](training-analysis-2026-10-28.md)，特征布局见 [model-inputs.md](model-inputs.md)。下文旧 run 结果属于历史 baseline。


更新时间：`2026-07-31`

本文描述当前仓库已经落地的工程结构和使用方式，不再把早期规划中的文件、命令或模型结构写成既成事实。长期路线与尚未实现的方向见 [architecture-plan.md](architecture-plan.md)，当前训练结论与下一步见 [status.md](status.md)。

## 1. 当前工程边界

仓库已经具备：

- 天凤 XML 解析、标准化事件写出和回放校验。
- `turn` / `response` 合法候选生成、监督样本构建和按局切分。
- 状态与候选动作特征离线预编码。
- 单任务和共享状态编码器的多任务候选排序训练。
- 单机训练、PyTorch DDP、流式 Parquet 数据集、checkpoint 和指标归档。
- 离线评估、response 阈值校准、response/turn 错例抽样。
- checkpoint 推理内核和 FastAPI HTTP 接口。

当前仍未实现：

- `value` 样本、value head 和 value 训练；`configs/task/value.yaml` 只是预留配置。
- `replay-eval` 与 `infer-on-state` 的业务逻辑；两个子命令已注册，但只返回占位提示。
- API 侧根据局面重新生成并验证完整合法动作。
- 在线对局 session、平台适配器、自博弈和完整计分引擎。

## 2. 实际目录结构

```text
train_mj/
  mjai/
    api/                    # FastAPI app、路由、请求 schema、中间件
    data/
      builders/             # 事件、校验、样本和 manifest 构建
      datasets/             # Parquet 存储、流式/内存数据集和 collator
      schema/               # Event、TurnSample 数据对象
      tenhou/               # Tenhou XML parser 与副露解码
    eval/                   # 离线评估、分布式评估、校准和错例抽样
    features/               # 状态与候选动作特征编码
    models/
      heads/turn_head.py    # 通用候选打分 head；turn/response 均复用
      encoder.py            # 单任务状态/动作 MLP 编码器
      policy_net.py         # 单任务 turn/response policy
      multitask_policy_net.py
    rules/                  # 牌表示、回放状态、和牌检查、合法动作、校验
    serving/                # checkpoint 加载、推理、适配与输入校验
    training/               # 训练引擎、loss、指标和类别权重
    utils/                  # 配置、设备、路径、随机种子
    cli.py                  # 统一命令入口
  configs/
    data/ model/ task/ train/
    experiments/
    turn_local_smoke.yaml
    response_local_smoke.yaml
    turn_p800.yaml
    response_p800.yaml
    multitask_p800.yaml
    api_local.yaml
  tests/
    fixtures/tenhou/
    test_*.py
  docs/
  start.sh
  docker-compose.p800.yml
  docker-compose.p800.multitask.yml
```

`data/`、`outputs/` 和 `runs/` 是本地产物目录，已被 Git 忽略，不应提交原始牌谱、checkpoint 或机器相关路径。

## 3. 数据链路

当前主线为：

```text
Tenhou XML
  -> preprocess
  -> bronze_events/*.parquet
  -> validate-events
  -> build-dataset
  -> silver_samples/{turn,response}/*.parquet
  -> make-manifest
  -> manifests/{train,valid,test}.jsonl
  -> train / eval / calibration
```

### 3.1 事件 schema

`bronze_events` 的物理字段以 `mjai/data/datasets/event_store.py::EVENT_SCHEMA` 为准：

```text
game_id
kyoku_index
step_index
event_type
actor
raw_tag
payload_json
```

局数、分数、牌、副露等事件专属内容保存在 `payload_json`，不是独立 Parquet 列。

### 3.2 样本 schema

`silver_samples` 的物理字段以 `mjai/data/datasets/sample_store.py::SAMPLE_SCHEMA` 为准：

```text
sample_id
game_id
kyoku_index
step_index
actor
decision_type
state_json
legal_actions_json
label_action_json
state_features_bin
candidate_features_bin
label_index
label_kind
```

关键约束：

- `label_index` 必须指向 `legal_actions_json` 中的真实标签动作。
- 特征在 `build-dataset` 阶段编码为 float32 字节串；训练和评估不应逐 epoch 重跑规则特征。
- 当前维度为 turn state/action `11969/89`、response state/action `12012/91`，详细布局见 [model-inputs.md](model-inputs.md)。
- 特征 schema、规则语义或 split 逻辑变化后必须重建 `silver_samples` 和 manifest；parser 或事件 schema 变化后必须从 `preprocess` 开始重建。

### 3.3 数据切分

`make-manifest` 对排序后的 `game_id` 计算 MD5 桶，使用固定的：

```text
train: 80%
valid: 10%
test: 10%
```

切分单位是完整对局，不会把同一 `game_id` 的决策样本分散到不同 split。小夹具集的实际局数可能因哈希桶取值偏离理论比例，并会输出空 split 或规模过小 warning。

## 4. 当前模型

### 4.1 单任务 policy

`MahjongPolicyNet` 是动态候选排序器：

```text
state_feats     -> 两层 MLP state encoder ----\
                                                -> pairwise feature -> MLP scorer -> candidate logits
candidate_feats -> 两层 MLP action encoder ---/
candidate_mask ---------------------------------> padding logits = -1e9
```

state 与 candidate 的联合特征由拼接、逐元素乘积和绝对差组成。当前实现没有 Transformer、CNN 或事件 token encoder。

### 4.2 多任务 policy

`MultitaskMahjongPolicyNet` 的真实共享边界为：

```text
turn state     -> turn projection     \
                                        -> shared state MLP -> turn head
response state -> response projection /

turn action     -> turn action MLP     -> turn head
response action -> response action MLP -> response head
```

turn/response 共享状态主干，但保留任务专属输入投影、action encoder 和候选打分 head。支持：

- `epoch_sequential`：每个 epoch 顺序训练两路任务。
- `batch_interleave`：按配置的有效 batch 数交错更新。

### 4.3 尚未消费的配置字段

`configs/model/encoder_small.yaml` 与 `encoder_base.yaml` 中的 `num_layers`、`num_heads`、`state_mlp_dim`、`action_mlp_dim` 和 `heads.value` 来自早期设计，目前训练引擎只消费 `hidden_size` 与 `dropout`。它们不能作为“Transformer/value 已实现”的证据，后续应选择删除、正式接线或迁入实验配置。

`grad_accum_steps`、`val_every_n_steps` 与 `save_every_n_steps` 目前也没有进入训练控制流；当前训练按 epoch 验证和保存。配置中出现这些字段不表示梯度累积或 step 级验证/保存已经生效。

## 5. 配置入口

| 场景 | 配置 |
| --- | --- |
| 本地 turn smoke | `configs/turn_local_smoke.yaml` |
| 本地 response smoke | `configs/response_local_smoke.yaml` |
| P800 turn 单任务 | `configs/turn_p800.yaml` |
| P800 response 单任务 | `configs/response_p800.yaml` |
| P800 多任务 DDP 主线 | `configs/multitask_p800.yaml` |
| v1.2 batch interleave 复现 | `configs/experiments/multitask_p800_v1_2_interleave.yaml` |
| 本地 HTTP API | `configs/api_local.yaml` |

配置加载器支持递归字典覆盖、`defaults` 合并和对顶层标量的 `${...}` 字符串插值；它不是完整 Hydra/OmegaConf。`defaults` 路径相对主配置所在目录解析，因此 `configs/experiments/` 下的配置若直接使用现有 `defaults` 写法会寻找错误的子目录。`turn_local_smoke.yaml` 与 `response_local_smoke.yaml` 是可直接运行的完整配置；分层的 `data/model/task/train` YAML 主要由 `configs/` 根目录下的 P800 主配置组合。

训练入口实际支持 `task.task: turn|response|multitask`。若传入 `value`，训练引擎会明确报错。

## 6. CLI 现状

统一入口：

```bash
python -m mjai.cli <subcommand> [args]
```

| 子命令 | 状态 | 用途 |
| --- | --- | --- |
| `preprocess` | 可用 | XML 转标准化事件分片 |
| `validate-events` | 可用 | 回放并输出事件校验报告 |
| `build-dataset` | 可用 | 构建 turn/response 样本；支持多进程和跳过重复校验 |
| `make-manifest` | 可用 | 生成 train/valid/test manifest |
| `train` | 可用 | 单任务或多任务训练 |
| `eval` | 可用 | 单任务 checkpoint 或多任务指定 head 的离线评估 |
| `calibrate-response` | 可用 | 扫描 pass bias 与无役鸣牌惩罚 |
| `sample-response-errors` | 可用 | 按 label→prediction confusion 抽样 |
| `sample-turn-errors` | 可用 | 按标签类型与真实动作排名抽样 |
| `inspect-sample` | 可用 | 按索引或 sample_id 查看样本 |
| `dump-case` | 可用 | 按 game/kyoku/step 回溯事件上下文 |
| `serve-api` | 可用 | 启动 FastAPI 服务 |
| `replay-eval` | 占位 | 已注册参数，业务逻辑未实现 |
| `infer-on-state` | 占位 | 已注册参数，业务逻辑未实现 |

占位命令当前会打印 `implementation pending` 并返回成功码，因此不能只根据退出码判断功能已经执行。

## 7. 常用工作流

### 7.1 本地 fixture 数据闭环

```bash
python -m mjai.cli preprocess \
  --input tests/fixtures/tenhou \
  --output data/bronze_events

python -m mjai.cli validate-events \
  --input data/bronze_events \
  --report outputs/reports/event_validation.json

python -m mjai.cli build-dataset \
  --input data/bronze_events \
  --output data/silver_samples \
  --workers 8 \
  --skip-validation

python -m mjai.cli make-manifest \
  --input data/silver_samples \
  --output data/manifests
```

2026-07-31 对当前 21 个 XML 夹具重新实测的结果为：

- `7386` 条事件，`21/21` 分片校验通过。
- `3532` 条 turn 样本，`10506` 条 response 样本。
- manifest 为 `19/1/1` 个 train/valid/test game。

`--skip-validation` 只应在同一批事件已经通过 `validate-events` 后使用。

### 7.2 训练与评估

```bash
python -m mjai.cli train --config configs/turn_local_smoke.yaml
python -m mjai.cli train --config configs/response_local_smoke.yaml

python -m mjai.cli eval \
  --ckpt outputs/checkpoints/turn_local_smoke/best.pt \
  --split valid \
  --report outputs/checkpoints/turn_local_smoke/eval_valid.json
```

多任务 checkpoint 评估时必须选择 head：

```bash
python -m mjai.cli eval --ckpt <multitask-ckpt> --task turn --split test
python -m mjai.cli eval --ckpt <multitask-ckpt> --task response --split test
```

### 7.3 调试与校准

```bash
python -m mjai.cli inspect-sample --dataset data/silver_samples --index 0
python -m mjai.cli dump-case --game-id <game_id> --kyoku-id <kyoku_id> --step-id <step_id>

python -m mjai.cli calibrate-response \
  --ckpt <response-ckpt> \
  --split valid \
  --manifest data/manifests/valid.jsonl \
  --report outputs/reports/response_calibration_valid.json
```

完整参数和错例抽样示例见 [smoke-training.md](smoke-training.md)。

### 7.4 P800/Linux

`start.sh` 可以串联 preprocess、validate、build、manifest、train、eval 和 calibration：

```bash
./start.sh
TASKS=turn ./start.sh
TASKS=response ./start.sh
TASKS=multitask DDP_NPROC_PER_NODE=8 ./start.sh
CLEAN_DATA=1 RUN_CALIBRATION=1 ./start.sh
```

`train.distributed: true` 时必须由 `torchrun` / `python -m torch.distributed.run` 启动。多任务 DDP 还要求 `find_unused_parameters: true`。训练 shard 按 rank 和 DataLoader worker 切分；验证、checkpoint 和指标文件只由 rank 0 写出。

## 8. 训练产物与评估

单任务训练会写出：

```text
config.resolved.yaml
epoch_metrics.jsonl
train_metrics.json
best.pt
best_accuracy.pt
best_loss.pt
best_macro.pt
final.pt
```

response 还会维护按 `non_pass_f1 - false_call_rate` 选择的 `best_response.pt`。多任务训练另外写出 `best_turn.pt`、`best_response.pt` 和 `best_multitask.pt`。

离线评估包含 top-k、macro top-k、动作类别聚合、候选规模分桶和有限错例。response 额外报告 pass/non-pass precision、recall、F1、false call、missed call 和 confusion。当前默认 serving 模型与 operating point 以 [status.md](status.md) 和 `configs/api_local.yaml` 为准。

## 9. Serving/API 边界

启动：

```bash
python -m mjai.cli serve-api --config configs/api_local.yaml
```

已实现：

- `GET /health`
- `GET /v1/models`
- `POST /v1/predict/turn`
- `POST /v1/predict/response`
- 可选 API key、统一错误响应、请求/响应 debug logging
- response `pass_bias` 和 `no_yaku_call_penalty`

调用方必须传入 `legal_actions`。服务端当前只校验 schema、必要字段、动作类型、座位/牌值范围和候选数量，不会根据 `state` 用规则引擎重新证明每个候选动作合法。完整边界见 [serving-api-plan.md](serving-api-plan.md)。

`configs/api_local.yaml` 是开发配置：默认允许无 API key，并打开请求/响应 body 日志；不应原样用于公网。`request_timeout_seconds` 当前也尚未接入实际超时控制。

## 10. 测试与变更约束

```bash
python -m pytest
python -m compileall -q mjai tests
git diff --check
```

2026-07-31 当前测试树可收集 `134` 个 case。全量测试包含多次 fixture 数据构建和 smoke train，耗时明显高于纯单元测试。

修改时至少同步以下内容：

| 变化 | 必须同步 |
| --- | --- |
| state/action 特征维度或语义 | `docs/model-inputs.md`、重建 silver/manifest、重训 checkpoint |
| parser 或 Event schema | 从 bronze 开始重建、更新 smoke/status 证据 |
| 合法动作或样本标签 | `docs/rule-coverage.md`、对应规则/样本测试 |
| CLI 参数或子命令 | 本文、`README.md`、`docs/smoke-training.md` |
| serving 请求或默认 operating point | `docs/serving-api-plan.md`、`configs/api_local.yaml`、`docs/status.md` |
| 新训练 run | `runs/<date>_<exp>/` 证据与对应训练分析文档 |

## 11. 当前工程债与下一步

优先级最高的工程事实是：

1. 当前牌力主线已经不是“先把训练跑起来”，而是复核 response 极保守 operating point、分析 turn 防铳与少数类回退。
2. 配置仍保留未消费的 Transformer/value 字段，应避免继续形成“配置看似支持、代码实际忽略”的歧义。
3. `replay-eval`、`infer-on-state` 的占位命令返回成功码，后续应实现或改为明确的非零退出。
4. API 合法候选仍由调用方负责，规则层和 serving 层尚未形成完整在线闭环。
5. 自动类别权重、分层采样、value/risk 辅助目标和完整计分仍是后续能力，不属于当前 baseline。

更具体的任务顺序见 [status.md](status.md)。


### 数据准备并行与性能调优

`preprocess` 和 `validate-events` 支持按文件多进程处理，默认仍为 `--workers 1`。
待执行任务数量有界，校验报告仍按文件名排序；串行与并行使用相同的解析和回放逻辑。
例如在 Linux 数据机器上：

```bash
python -m mjai.cli preprocess --input /workspace/dataset_100 --output data/full_state_v2/bronze_events --workers 32
python -m mjai.cli validate-events --input data/full_state_v2/bronze_events --report outputs/reports/full_state_v2_events.json --workers 32
# 确认报告 files_failed=0 后执行：
python -m mjai.cli build-dataset --input data/full_state_v2/bronze_events --output data/full_state_v2/silver_samples --workers 32 --skip-validation
```

这里的 workers 是进程数。多进程数据 worker 的 Arrow CPU/IO 线程池限制为各 1 个，
避免文件级并行叠加进程内部的大线程池。preprocess/validate-events 在 Windows 上最多使用
61 个进程。增加进程数不保证增加吞吐；使用相同的代表性输入，分别测试 32、64、96、128、180，
输出到各自的临时目录，以总耗时和 samples_per_s 比较，同时观察内存、swap、磁盘和 CPU。
手牌判定中多个 LRU 缓存是每进程独立的，缓存随处理量增长，短测不能代表长任务的内存峰值。

候选特征编码在每个样本内复用公共可见牌计数，保留被鸣牌的两种计数约定，并在加上
候选手牌及额外可见牌后才截断到 4。float32 写出直接使用 array 构造器，减少 Python
逐元素转换；上述修改不改变 full_state_v2 特征布局或样本内容。

本地 21 局 fixture（14,038 个样本）一次单进程、无 profiler 的对比：修改前 62.83 秒、
223.4 samples/s，修改后 34.01 秒、412.8 samples/s。同期有测试进程运行，结果仅为本地
参考，不代表 P800 的高并发加速比。42 个输出分片所有列（包含特征字节）逐项比较一致。
