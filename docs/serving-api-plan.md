# Serving API 方案

更新时间：`2026-06-01`

本文记录为项目新增 API 模块的方案与当前实现状态。目标是让外部应用可以通过稳定接口调用当前 `turn` / `response` 模型，同时保持规则、特征、模型和 HTTP 服务之间的边界清晰。

当前状态：阶段 1 与阶段 2 已落地到代码，`mjai/serving/` 提供纯 Python 推理内核，`mjai/api/` 提供 FastAPI 薄封装，`python -m mjai.cli serve-api --config configs/api_local.yaml` 已注册为启动入口。阶段 3 的服务端合法动作生成和阶段 4 的在线对局 session 层仍保留为后续工作。

## 1. 目标与边界

第一版 API 只解决“给定局面与合法候选动作，返回模型排序结果”。

核心目标：

- 复用现有 `mjai/features/`、`mjai/models/`、`mjai/eval/` 的模型输入和 checkpoint 加载逻辑。
- 模型只在 `legal_actions` 候选集合内打分，最终输出必须来自合法候选集合。
- `response` 推理显式接入当前已验证的 `pass_bias` operating point，默认使用 v1.0.0 improved response 的 `pass_bias=0.3`。
- `response` 推理支持 `no_yaku_call_penalty`，用于对“首次开门后无明显役路”的 `chi/pon/daiminkan` 候选额外扣分，降低无役鸣牌 false call。
- HTTP 层保持薄封装，业务推理能力沉到 `mjai/serving/`，便于 CLI、测试和未来在线对局代理复用。

第一版不做：

- 不做完整在线对局代理。
- 不维护长连接游戏 session。
- 不让模型直接生成任意动作。
- 不把合法动作生成、模型加载、HTTP 请求处理混在同一个模块。
- 不开放请求侧传入任意 checkpoint 路径。

## 2. 已落地目录结构

```text
mjai/
  serving/
    __init__.py
    checkpoint.py        # 加载 checkpoint/config，并构建 turn/response/multitask 模型
    predictor.py         # 纯 Python 推理入口：state + legal_actions -> ranked actions
    schemas.py           # serving 层 dataclass / TypedDict
    action_adapter.py    # logits -> 可执行动作；response pass_bias 策略放这里
    validation.py        # 请求内核校验、候选动作合法性兜底
  api/
    __init__.py
    app.py               # FastAPI app factory 和启动入口
    deps.py              # API key middleware 与安全配置
    routes.py            # /health /metadata /predict 路由
    schemas.py           # Pydantic 请求/响应模型
configs/
  api_local.yaml
tests/
  test_serving_predictor.py
  test_api_routes.py
```

`mjai/serving/` 是核心模块，不能依赖 FastAPI。`mjai/api/` 可以依赖 FastAPI 和 Pydantic，但只能调用 serving 层公开接口。

## 3. 模块职责与实现状态

### `mjai/serving/checkpoint.py`

职责：

- `torch.load` checkpoint。
- 读取 checkpoint 内的 `config`。
- 根据 `task.task` 构建 `MahjongPolicyNet` 或 `MultitaskMahjongPolicyNet`。
- 加载 `model_state_dict`。
- 设置 `eval()`、device、dtype 策略。
- 输出模型元信息，例如 checkpoint path、task、hidden size、feature dim、默认 `pass_bias`。

已实现公开接口：

```python
def load_policy_checkpoint(
    path: str | Path,
    *,
    task: str | None = None,
    device: str | None = None,
) -> LoadedPolicy:
    ...
```

`LoadedPolicy` 应包含：

- `model`
- `config`
- `checkpoint_path`
- `checkpoint_task`
- `device`
- `candidate_limit`
- response 默认 `pass_bias`
- response `no_yaku_call_penalty`

### `mjai/serving/predictor.py`

职责：

- 接收 `task`、`state`、`legal_actions`。
- 调用 `encode_turn_state` / `encode_response_state`。
- 调用 `encode_turn_candidate_actions` / `encode_response_candidate_actions`。
- 构造单样本 batch：

```text
state_feats: [1, state_dim]
candidate_feats: [1, C, action_dim]
candidate_mask: [1, C]
```

- 调用 `model.forward_turn` 或 `model.forward_response`。
- 返回排序后的候选动作、score、prob、rank。

已实现公开接口：

```python
class MahjongPredictor:
    def predict_turn(self, state: dict, legal_actions: list[dict], *, top_k: int = 5) -> PredictionResult:
        ...

    def predict_response(
        self,
        state: dict,
        legal_actions: list[dict],
        *,
        top_k: int = 5,
        pass_bias: float | None = None,
    ) -> PredictionResult:
        ...
```

### `mjai/serving/action_adapter.py`

职责：

- 将 logits 转为排序结果。
- 对 `turn` 直接按 logits 排序。
- 对 `response` 支持两种口径：
  - `raw`：直接按 logits 排序。
  - `calibrated`：使用 `pass_bias` 选择 `pass` 或最佳非 `pass`。
  - `safe calibrated`：在 calibrated 前先对无明显开副露役路的 call 候选应用 `no_yaku_call_penalty`。

`response` calibrated 选择规则必须和现有 calibration 逻辑一致：

```text
choose_non_pass = non_pass_exists and (
  not pass_exists or best_non_pass_score - best_pass_score > pass_bias
)
```

当 `no_yaku_call_penalty > 0` 且候选特征中的 response call/context `features[38] == 1` 时，先从该 `chi/pon/daiminkan` 候选 logit 中扣除该惩罚，再执行上面的 `pass_bias` 比较。raw prediction 与 top-k 排序仍保留模型原始 logits，用于诊断模型本身的偏好。

离线诊断命令也使用同一选择口径：`calibrate-response` 支持 `--no-yaku-call-penalties` / `--no-yaku-call-penalty` 扫描或固定无役候选扣分，并在报告中写入 `no_yaku_false_call_rate`、`no_yaku_false_call_share`；`sample-response-errors` 支持 `--no-yaku-call-penalty`，用于按同一安全口径抽取 `pass->chi` / `pass->pon` 等错例。

当前默认值：

```text
default:      pass_bias = 0.3
recall:       pass_bias = 0.16
conservative: pass_bias = 0.4
no_yaku_call_penalty = 0.4  # configs/api_local.yaml
```

### `mjai/serving/validation.py`

职责：

- 校验 `legal_actions` 非空。
- 校验候选动作数量不超过配置的 `candidate_limit`。
- 校验 `state` 至少包含特征编码所需字段。
- 校验 `response` 请求包含 `response_target.source` 和 `response_target.tile`。
- 确保输出动作一定是输入 `legal_actions` 中的一项。

第一版已实现结构与基础值校验，不要求从完整 `GameState` 重新生成合法动作。当前校验范围包括：

- `legal_actions` 必须为非空列表，且不超过 `min(api.max_legal_actions, checkpoint_candidate_limit)`。
- `top_k` 必须为正数，`response.pass_bias` 必须为非负数。
- `state.hand`、`all_discards`、`own_melds`、`all_melds`、`dora_indicators`、`scores` 等字段会按推理特征需要归一化。
- `response` 请求必须提供 `state.round_wind`，并按 `0..3` 归一化。它不进入 state 向量维度，但会影响 response 役牌路线与无役风险候选特征。
- `response` 请求必须包含 `state.response_target.source` 与 `state.response_target.tile`。
- tile 字符串、actor/source 座位、action kind 会做基础合法性检查。
- `response` 请求缺失 `all_melds` 时归一为 `[[], [], [], []]`。

后续阶段再支持服务端调用规则引擎生成合法候选动作。

## 4. HTTP 接口设计

### `GET /health`

用于健康检查。

响应：

```json
{
  "status": "ok"
}
```

### `GET /v1/models`

返回当前加载的模型元信息。

响应示例：

```json
{
  "models": {
    "turn": {
      "checkpoint": "outputs/checkpoints/turn_local_smoke/best.pt",
      "checkpoint_task": "turn",
      "device": "cuda",
      "state_dim": 8173,
      "action_dim": 89
    },
    "response": {
      "checkpoint": "outputs/checkpoints/response_local_smoke/best.pt",
      "checkpoint_task": "response",
      "device": "cuda",
      "state_dim": 8216,
      "action_dim": 91,
      "default_pass_bias": 0.3
    }
  }
}
```

### `POST /v1/predict/turn`

输入 `turn` 决策局面和合法动作，返回候选排序。

请求示例：

```json
{
  "state": {
    "actor": 0,
    "hand": {"1m": 1, "2m": 1, "3m": 1},
    "all_discards": [[], [], [], []],
    "last_discard_tile": null,
    "last_discard_actor": null,
    "riichi_declared": false,
    "temporary_furiten": false,
    "riichi_furiten": false,
    "own_melds": [],
    "dealer": 0,
    "seat_wind": 0,
    "dora_indicators": [],
    "scores": [25000, 25000, 25000, 25000],
    "turn_number": 1,
    "remaining_draws_estimate": 69,
    "honba": 0,
    "riichi_sticks": 0
  },
  "legal_actions": [
    {"kind": "discard", "tile": "1m"},
    {"kind": "discard", "tile": "2m"},
    {"kind": "discard", "tile": "3m"}
  ],
  "top_k": 5
}
```

响应示例：

```json
{
  "task": "turn",
  "selection_mode": "raw",
  "selected_action": {"kind": "discard", "tile": "2m"},
  "predictions": [
    {
      "rank": 1,
      "action": {"kind": "discard", "tile": "2m"},
      "score": 1.23,
      "prob": 0.72
    },
    {
      "rank": 2,
      "action": {"kind": "discard", "tile": "1m"},
      "score": 0.28,
      "prob": 0.28
    }
  ],
  "model": {
    "checkpoint": "outputs/checkpoints/turn_local_smoke/best.pt",
    "checkpoint_task": "turn"
  }
}
```

### `POST /v1/predict/response`

输入 `response` 决策局面和合法动作，返回 raw 排序与 calibrated 选择。

请求示例：

```json
{
  "state": {
    "actor": 1,
    "hand": {"1m": 1, "2m": 1, "4m": 1},
    "all_discards": [[], [], [], ["3m"]],
    "last_discard_tile": "3m",
    "last_discard_actor": 3,
    "riichi_declared": false,
    "temporary_furiten": false,
    "riichi_furiten": false,
    "own_melds": [],
    "dealer": 0,
    "seat_wind": 1,
    "dora_indicators": [],
    "scores": [25000, 25000, 25000, 25000],
    "turn_number": 5,
    "remaining_draws_estimate": 55,
    "honba": 0,
    "riichi_sticks": 0,
    "response_target": {"source": 3, "tile": "3m"}
  },
  "legal_actions": [
    {"kind": "pass"},
    {"kind": "chi", "tile": "3m", "consumed_tiles": ["1m", "2m"], "source": 3},
    {"kind": "chi", "tile": "3m", "consumed_tiles": ["2m", "4m"], "source": 3}
  ],
  "top_k": 5,
  "pass_bias": 0.3
}
```

响应示例：

```json
{
  "task": "response",
  "selection_mode": "calibrated",
  "pass_bias": 0.3,
  "selected_action": {"kind": "pass"},
  "raw_selected_action": {
    "kind": "chi",
    "tile": "3m",
    "consumed_tiles": ["2m", "4m"],
    "source": 3
  },
  "predictions": [
    {
      "rank": 1,
      "action": {
        "kind": "chi",
        "tile": "3m",
        "consumed_tiles": ["2m", "4m"],
        "source": 3
      },
      "score": 0.81,
      "prob": 0.46
    },
    {
      "rank": 2,
      "action": {"kind": "pass"},
      "score": 0.62,
      "prob": 0.38
    }
  ],
  "model": {
    "checkpoint": "outputs/checkpoints/response_local_smoke/best.pt",
    "checkpoint_task": "response"
  }
}
```

## 5. 配置设计

已新增 `configs/api_local.yaml`：

```yaml
api:
  host: 0.0.0.0
  port: 8000
  enable_docs: true
  request_timeout_seconds: 5
  max_legal_actions: 64
  debug_log_requests: false
  debug_log_responses: true
  debug_log_max_body_chars: 20000
  debug_log_logger: uvicorn.error

device: cuda

models:
  turn:
    checkpoint: outputs/checkpoints/turn_local_smoke/best.pt
  response:
    checkpoint: outputs/checkpoints/response_local_smoke/best.pt
    pass_bias:
      default: 0.3
      recall: 0.16
      conservative: 0.4

security:
  api_key_env: MJAI_API_KEY
  allow_without_api_key: true
```

部署或对外暴露时，应将 `allow_without_api_key` 改为 `false`，并通过环境变量注入 `MJAI_API_KEY`。

`configs/api_local.yaml` 中的 checkpoint 路径是本地示例路径。启动服务前必须先确认 `models.turn.checkpoint` 与 `models.response.checkpoint` 指向真实存在的 checkpoint。

调试前端对接时可以临时打开 `api.debug_log_requests: true`。开启后，API 会把
`POST /v1/predict/turn` 与 `POST /v1/predict/response` 的原始 JSON 请求体写入
`api.debug_log_logger` 指定的 logger；`debug_log_responses` 默认为 `true`，会同时输出
预测返回体，包含 `selected_action`、`predictions` 等字段。请求体和响应体都默认最多保留
`debug_log_max_body_chars` 个字符。该开关默认关闭；线上或对外暴露服务不要长期打开，避免泄露牌局数据或产生过量日志。

## 6. 依赖管理

API 依赖应放到 optional dependency，避免训练环境默认安装 Web 依赖：

```toml
[project.optional-dependencies]
api = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
]
```

本地安装：

```bash
pip install -e ".[api]"
```

## 7. 启动方式

推荐新增 CLI 子命令：

```bash
python -m mjai.cli serve-api --config configs/api_local.yaml
```

也可以提供模块入口：

```bash
python -m mjai.api.app --config configs/api_local.yaml
```

如果用 `uvicorn` 直接启动，建议通过 app factory 注入配置：

```bash
set MJAI_API_CONFIG=configs/api_local.yaml
uvicorn mjai.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

PowerShell 可以使用：

```powershell
$env:MJAI_API_CONFIG="configs/api_local.yaml"
uvicorn mjai.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

直接 `uvicorn` 需要通过 `MJAI_API_CONFIG` 传项目配置；本地优先使用 CLI 启动入口。

## 8. 错误码约定

建议 HTTP 错误语义：

| 状态码 | 场景 |
|---:|---|
| `400` | 请求 JSON 结构错误、`legal_actions` 为空、`top_k` 非法 |
| `401` | API key 缺失或错误 |
| `413` | 候选动作数量超过 `max_legal_actions` |
| `422` | state 字段缺失、tile/actor/action kind 非法 |
| `500` | 模型加载失败、推理异常 |
| `503` | 服务启动中或模型未加载 |

错误响应格式：

```json
{
  "error": {
    "code": "invalid_request",
    "message": "legal_actions must not be empty"
  }
}
```

## 9. 测试计划

### serving 层单测

重点覆盖：

- 从 smoke checkpoint 加载 turn / response 模型。
- 使用现有 `silver_samples` 或测试夹具中的 `state_json`、`legal_actions_json` 做单样本推理。
- 返回的 `selected_action` 必须存在于输入 `legal_actions`。
- `top_k` 截断正确。
- `response pass_bias=0.0` 与 raw argmax 一致。
- `response pass_bias=0.3` 使用 calibrated 选择逻辑。

### API 层测试

重点覆盖：

- `GET /health`。
- `GET /v1/models`。
- `POST /v1/predict/turn` 成功返回。
- `POST /v1/predict/response` 成功返回。
- 空 `legal_actions` 返回 `400`。
- 超过 `max_legal_actions` 返回 `413`。
- 开启 API key 后，未携带 key 返回 `401`。

### 回归验收

最低验收标准：

- `pytest tests/test_serving_predictor.py tests/test_api_routes.py tests/test_cli.py::test_cli_registers_serve_api_command` 通过。
- 在 `configs/api_local.yaml` 指向真实 checkpoint 后，`python -m mjai.cli serve-api --config configs/api_local.yaml` 可启动。
- 用一条 turn 样本和一条 response 样本请求 API，返回动作都来自输入候选集合。
- response 默认 `pass_bias=0.3` 在配置和响应中都可追踪。
- 结构错误的请求返回 `400`，业务校验错误返回 `422` 或 `413`，API key 缺失返回 `401`。

## 10. 分阶段落地

### 阶段 1：纯 serving 推理内核

状态：已完成。

实现：

- `mjai/serving/checkpoint.py`
- `mjai/serving/predictor.py`
- `mjai/serving/action_adapter.py`
- `tests/test_serving_predictor.py`

验收：

- 不启动 HTTP 服务，也能用 Python 调用完成 turn / response 推理。
- 单样本 batch 构造和离线 eval 的输入张量维度一致。

### 阶段 2：FastAPI 薄封装

状态：已完成。

实现：

- `mjai/api/app.py`
- `mjai/api/routes.py`
- `mjai/api/schemas.py`
- `configs/api_local.yaml`
- `tests/test_api_routes.py`

验收：

- 本地 API 可启动。
- `/health`、`/v1/models`、`/v1/predict/turn`、`/v1/predict/response` 可用。

### 阶段 3：合法动作服务端生成

在第一版稳定后，再支持客户端只传局面，由服务端调用规则引擎生成合法候选：

- `turn`：基于 `HandState`、`GameState` 调用 `legal_turn_actions`。
- `response`：基于 `HandState`、`GameState`、`response_target` 调用 `legal_responses`。

该阶段需要更严格定义外部局面 schema，因为当前训练样本的 `state_json` 是特征编码视角的快照，不等价于完整可回放 `GameState`。

状态：未开始。

### 阶段 4：在线对局代理

后续若需要接入实时应用，再新增 session 层：

- 维护牌局事件流。
- 增量更新 `GameState`。
- 在自摸/响应窗口调用 serving predictor。
- 输出可执行动作。

这部分应作为单独模块设计，不应混入第一版 API。

状态：未开始。

## 11. 风险与注意事项

- 特征 schema 已在 `docs/model-inputs.md` 固化；只要 state/action feature 维度变化，API 推理必须和训练数据同步更新。
- `response` 的默认模型应优先使用 2026-05-10 improved 单任务 response checkpoint，而不是 v1.1 multitask response。
- 外部应用如果自行生成 `legal_actions`，必须和项目规则引擎动作格式保持一致。
- API 服务启动时应一次性加载模型，不应每个请求重新加载 checkpoint。
- GPU 推理应使用单模型常驻；如果后续做高并发，需要再评估 batching、队列和多进程模型副本。
- 对外暴露时必须加 API key 或放在可信内网后面，不要开放任意 checkpoint 路径或本地文件路径。
