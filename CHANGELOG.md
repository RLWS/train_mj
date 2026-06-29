# Changelog

## v1.1.0 - 2026-05-17

本版本标记当前工程进入“turn / response 初步可用，HTTP 推理接口可用”的发布点。它基于 `v1.0.0` 的 P800 单任务 improved baseline，新增多任务训练主线、第一版 serving/API、更多离线评估和错例抽样工具。

### Added

- 新增 `MultitaskMahjongPolicyNet` 和 `task: multitask` 训练入口，支持 `turn` / `response` 共享状态 encoder，并保留任务专属 state projection、action encoder 和 ranking head。
- 新增 P800 多任务训练配置与记录：默认主线回到 v1.1 epoch-sequential，多任务 v1.2 `batch_interleave` 作为显式实验配置保留在 `configs/experiments/`。
- 新增 serving/API 模块：`/health`、`/v1/models`、`/v1/predict/turn`、`/v1/predict/response`。
- 新增 API key middleware、统一错误响应、请求/响应 debug logging，以及 `python -m mjai.cli serve-api --config configs/api_local.yaml` 启动入口。
- 新增 response 推理 `pass_bias` operating point 配置；当前默认已更新为 40w `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8`，并保留低 bias 对照档。
- 新增 `calibrate-response`、`sample-response-errors`、`sample-turn-errors` 等评估和错例抽样命令。

### Changed

- 包版本、`mjai.__version__` 和 FastAPI OpenAPI 版本同步为 `1.1.0`。
- 离线 eval 默认使用 streaming dataset，减少 valid/test split 常驻内存压力。
- response checkpoint 选择更关注 `non_pass_f1 - false_call_rate`，避免被 pass-only 或 overall accuracy 支配。
- `build-dataset` 输出预编码 `state_features_bin` 和 `candidate_features_bin`，训练和评估阶段不再逐 epoch 重算规则特征。
- 默认 P800 多任务入口保持 v1.1 路径；v1.2 interleave 只在需要复现实验或继续验证调度变量时显式使用。

### Validated

- `pytest` 全量通过：`114 passed`。
- `python -m compileall -q mjai tests` 通过。
- 夹具数据链路已验证：`20` 局 Tenhou XML 可完成 preprocess、validate-events、build-dataset、make-manifest、smoke train 和 eval。
- P800 20 万局数据上已完成单任务 improved、多任务 v1.1 与 v1.2 interleave 对照评估。

### Known Limitations

- `response` 当前默认模型已切到 2026-06-24 40w v1.1 DDP `best_response.pt`，默认 `pass_bias=2.0 + no_yaku_call_penalty=0.8`，优先压低错误鸣牌；v1.0.0 improved 单任务 checkpoint 继续作为历史 baseline。
- API 第一版要求调用方传入 `legal_actions`；服务端合法动作生成、在线对局 session 和完整代理尚未实现。
- 规则引擎仍是 first-pass 版本，不是完整日麻规则/计分引擎；部分和牌机会放弃语义、途中流局、立直边界、多家荣和等仍需后续补齐。
- DDP、value head 和自动类别权重/分层采样尚未纳入默认训练主线。
- release 不包含本地 `data/`、`outputs/`、`runs/`、checkpoint 或原始牌谱文件。

完整发布说明见 [docs/releases/v1.1.0.md](docs/releases/v1.1.0.md)。
