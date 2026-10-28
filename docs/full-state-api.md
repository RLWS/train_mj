# full_state_v2 推理接口

在本分支仓库根目录运行：

```powershell
python -m mjai.cli serve-api --config configs/api_full_state.yaml
Invoke-RestMethod http://127.0.0.1:8001/v1/models
```

默认加载 `outputs/checkpoints/multitask_full_state_v2_ddp/best_turn.pt` 和 `best_response.pt`。服务读取 checkpoint 中的模型配置创建网络，并校验特征版本；旧版本权重不能混用。客户端只提交 JSON 状态和合法候选，不需要自行生成特征向量或配置网络宽度。

`GET /v1/models` 与两个预测端点响应中的 `model` 对象均返回 `feature_schema`、`state_dim`、`action_dim`、`hidden_size` 等实际模型信息。当前特征版本为 `full_state_v2`，turn 的 state/action 维度为 11969/89，response 为 12012/91。

`POST /v1/predict/turn` 和 `/v1/predict/response` 接收 `state`、`legal_actions` 和可选 `top_k`。response 可选 `pass_bias`；省略时使用后端配置，显式 0 仍表示覆盖。响应校准还包含服务端的 `no_yaku_call_penalty`。

状态必须包含 `feature_schema` 和完整公开状态；缺失字段不会静默退回旧编码。分数用实际点数（例如 25000），座位保持平台绝对座位，特征编码器统一旋转相对视角。公开事件的 `discard_count` 必须非负、按时间递增或相等，且不超过弃牌历史长度。事件中提供的 actor/source/tile 即使为可选字段也会校验；非法状态返回 HTTP 422 / `invalid_state`。

客户端还需要保留完整全局弃牌历史、四家副露、摸牌计数与立直确认时点。杠事件应先于新宝牌事件。抢杠窗口的公开状态取宣杠前快照；加杠事件的来源和叫牌沿用原碰。

已知训练侧边界：采样器的抢杠目标仍取原碰叫牌；当实际加杠牌与原叫牌的红五属性不同时，精确目标特征存在差异。客户端必须保留实际加杠牌保证合法候选正确；训练采样修复与数据重建需另行处理。

验证命令：

```powershell
python -m pytest tests/test_api_routes.py tests/test_serving_validation.py tests/test_serving_predictor.py -q
```

客户端目录中的 `tests/test_full_state_adapter.py` 可通过环境变量 `TRAIN_MJ_ROOT` 指向本仓库，加载上述真实权重进行 CPU 联调。测试使用 FastAPI TestClient，不占用监听端口、不触发游戏操作。
