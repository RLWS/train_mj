# Changelog

## Unreleased

### Documentation

- 按 2026-07-31 代码审计更新当前架构、工程布局、规则边界、CLI 状态和 fixture 基线。
- 明确当前模型是 MLP candidate ranker，value、`replay-eval`、`infer-on-state` 与服务端合法动作生成尚未实现。
- 记录 Git tag 与 Python 包版本元数据不一致。

## v1.2.0 - 2026-06-29

该标签覆盖从 v1.1.0 到 40w DDP、分布式评估/校准和极保守 response 默认档的工程主线。

### Added

- PyTorch DDP 训练和 streaming dataset rank/worker 分片。
- 分布式 offline eval 与 response calibration。
- response `no_yaku_call_penalty`、无役 false-call 指标与联合阈值扫描。
- 40w 多任务 v1.1 DDP 训练、评估和 calibration 证据。

### Changed

- 修正 turn discard 等待牌特征和 response 场风役牌路线。
- 稳定数据构建 worker，并确保 DDP 各 rank 执行输入校验。
- response serving 默认切到 40w `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8`。

### Known Limitations

- 极保守 response 默认档显著增加漏鸣，需要继续实战复核。
- API 要求调用方提供 `legal_actions`。
- value、`replay-eval`、`infer-on-state` 和完整在线代理未实现。
- Git tag 是 `v1.2.0`，但 `pyproject.toml`、`mjai.__version__` 与 OpenAPI 仍为 `1.1.0`。

完整说明见 [docs/releases/v1.2.0.md](docs/releases/v1.2.0.md)。

## v1.1.0 - 2026-05-17

该版本是 2026-05-17 的历史发布快照：turn/response 初步可用，多任务训练和第一版 HTTP 推理接口落地。

### Added

- `MultitaskMahjongPolicyNet` 和 `task: multitask` 训练入口。
- v1.1 epoch-sequential 与 v1.2 batch-interleave 实验配置。
- `/health`、`/v1/models`、`/v1/predict/turn`、`/v1/predict/response`。
- API key middleware、统一错误响应和请求/响应 debug logging。
- response `pass_bias` calibration 与 response/turn 错例抽样。
- 数据集 state/action 特征离线预编码。

### Validated at Release

- `pytest`：`114 passed`。
- `python -m compileall -q mjai tests` 通过。
- 当时的 20 个 Tenhou XML 夹具可完成 preprocess、validate、build、manifest、smoke train 和 eval。
- P800 20w 数据上完成单任务 improved、多任务 v1.1 与 v1.2 interleave 对照。

### Historical Limitations

- 发布时 DDP 尚未进入默认主线；该项已在 v1.2.0 完成。
- API 要求调用方提供 `legal_actions`。
- value、`replay-eval`、`infer-on-state` 和完整在线代理未实现。

冻结发布说明见 [docs/releases/v1.1.0.md](docs/releases/v1.1.0.md)。
