# full_state_v2 推理接口

在仓库根目录运行：

```powershell
python -m mjai.cli serve-api --config configs/api_local.yaml
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

默认加载 `runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/` 下的 `best_turn.pt` 和 `best_response.pt`。服务读取 checkpoint 中的模型配置创建网络，并校验特征版本；旧版本权重不能混用。客户端只提交 JSON 状态和合法候选，不需要自行生成特征向量或配置网络宽度。

`GET /v1/models` 与两个预测端点响应中的 `model` 对象均返回 `feature_schema`、`state_dim`、`action_dim`、`hidden_size` 等实际模型信息。当前特征版本为 `full_state_v2`，turn 的 state/action 维度为 11969/89，response 为 12012/91。

`POST /v1/predict/turn` 和 `/v1/predict/response` 接收 `state`、`legal_actions` 和可选 `top_k`。response 可选 `pass_bias`；省略时使用后端配置，显式 0 仍表示覆盖。响应校准还包含服务端的 `no_yaku_call_penalty`。当前默认 operating point 为 `pass_bias=1.5`、`no_yaku_call_penalty=0`（valid 网格 non-pass F1 最优，见 [最终分析](training-analysis-2026-10-28.md)）。

## 请求字段契约

- `feature_schema: "full_state_v2"`；缺失必需字段返回 HTTP 422 / `invalid_state`，不会静默退回旧编码或用零值补齐。
- `scores` 为实际点数（例如 25000）；actor、dealer、source 及四家数组仍为平台绝对座位，特征编码器统一旋转相对视角。
- `round_wind`、`round_number` 均为 0..3，后者为一至四局，不能把连庄次数算进局号。
- `match_rounds` 为 4（东风）、8（半庄）或 null（确实未知）。
- `remaining_tiles` 为四麻剩余摸牌机会，初始 70；包含岭上摸牌的消耗。
- `all_melds` 必须有四家（含赤牌、来源、暗杠和加杠后的组成），`own_melds` 必须与 `all_melds[actor]` 一致。
- `discard_history` 每项包含 actor、tile、tsumogiri（布尔或 null）、riichi_discard（布尔）。
- `public_events` 保存 reach1/reach2/chi/pon/daiminkan/ankan/kakan/dora；每项带 discard_count。reach 需 actor；鸣牌需 actor/source/tile；dora 需 tile。discard_count 必须非负、按时间非递减、且不超过弃牌历史长度。
- 必须提供四家立直标记/巡目、牌河、宝牌指示牌、手牌、本场、供托、自家实际摸牌次数等完整公开状态。
- turn 选择立直前的请求不能预先写入本次 reach1；对手已经发生的 reach1 则应保留。
- 杠事件应先于新宝牌事件。抢杠窗口的公开状态取宣杠前快照；加杠事件的来源和叫牌沿用原碰。
- 不提供对手隐藏手牌、摸牌牌面或未来事件。事件中提供的 actor/source/tile 即使为可选字段也会校验。

已知训练侧边界：采样器的抢杠目标仍取原碰叫牌；当实际加杠牌与原叫牌的红五属性不同时，精确目标特征存在差异。客户端必须保留实际加杠牌保证合法候选正确；训练采样修复与数据重建需另行处理。

完整向量布局见 [model-inputs.md](model-inputs.md)。

## 验证

```powershell
python -m pytest tests/test_api_routes.py tests/test_serving_validation.py tests/test_serving_predictor.py -q
```

客户端目录中的 `tests/test_full_state_adapter.py` 可通过环境变量 `TRAIN_MJ_ROOT` 指向本仓库，加载上述真实权重进行 CPU 联调。测试使用 FastAPI TestClient，不占用监听端口、不触发游戏操作。
