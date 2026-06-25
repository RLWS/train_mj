# 雀魂四麻 AI 训练仓库

本仓库用于构建基于天凤牌谱的雀魂四麻 AI 训练链路，目标是形成一条可验证、可训练、可扩展的工程闭环：

```text
天凤 XML -> 事件解析 -> 状态回放 -> 合法动作生成 -> 样本构建 -> 模型训练 -> 推理校验
```

## 当前状态

- 已完成第一阶段的数据闭环：天凤 XML 解析、`bronze_events` 持久化、回放校验、合法动作生成、`turn/response` 样本构建、Parquet 写出、按 `game_id` 的 manifest 切分。
- 已补上第一版诊断输出：事件分布、失败原因聚合、动作标签分布、split 样本数和空 split warning。
- 已打通第一版训练闭环：`turn/response` manifest 读取、状态/动作特征、候选排序模型、类别加权 loss、总体/宏平均/按动作类别指标、checkpoint 与指标落盘；2026-05-06 已有 P800 大数据单任务 baseline。
- 已补上第一版离线评估：可从 checkpoint 对 `turn/response` 的 `train/valid/test` split 做 `loss`、`top-k accuracy`、`macro top-k`、按动作类别与主类/非主类分组聚合评估，并导出 JSON 报告。
- 已补上第一版样本抽查：可直接从 `silver_samples` 目录按全局索引读取样本，查看 label、合法动作和状态快照。
- 已补上第一版 case 定位：可从 `eval` 错例跳转到 `inspect-sample`，再按 `game_id / kyoku_index / step_index` 回溯 `bronze_events` 上下文。
- 已补上第一版自家高价值回合动作：`turn` 合法动作与样本链路已支持 `tsumo`、`ankan`、`kakan` 和 `kyuushu_kyuuhai`，其中当前真实夹具已覆盖 `tsumo`、`ankan` 与 `kyuushu_kyuuhai` 样本。
- 已补上第一版 `ron` 正确性过滤：当前 `response` 合法动作与样本链路已支持自家舍牌振听、同巡振听和立直后振听。
- 已补上第一版 `chankan`：当前 `response` 合法动作、样本标签与训练动作编码已支持抢杠，且 `kakan` 响应窗口已收紧为只暴露 `pass / ron / chankan`。
- 已修正天凤 `daiminkan` 副露解码中的赤牌归属边界：明杠会使用 Tenhou anchor tile 作为被鸣牌，避免把来源低位误当作 called index。
- `build-dataset` 已支持 `--workers N` 按分片并行构建样本，适合在本机或训练机上扩大真实牌谱数据量。
- 已补上第一版 serving/API：`mjai/serving` 提供推理内核，`mjai/api` 提供 FastAPI 封装，支持 `/health`、`/v1/models`、`/v1/predict/turn` 和 `/v1/predict/response`。
- P800 20w/40w 数据上已完成单任务 improved、v1.2 safety 单任务、多任务 v1.1/v1.2 和 40w v1.1 DDP run 归档；当前 `turn` 主候选已更新为 40w `multitask/best_turn.pt`，`response` 稳定 serving baseline 仍保留 2026-05-10 improved，40w `best_response.pt + pass_bias=0.0 + no_yaku_call_penalty=0.4` 作为第一对照候选。
- 当前测试基线为 `pytest` 全量通过，结果是 `127 passed`。
- 使用 `tests/fixtures/tenhou` 实测可完成：
  - `20` 局 XML 预处理
  - `6281` 条事件写入 `bronze_events`
  - `20` 个事件分片全部校验通过
  - `2996` 条 `turn` 样本
  - `8910` 条 `response` 样本
  - `turn` 单任务 `1` epoch smoke train
  - `response` 单任务 `1` epoch smoke train
  - `turn` / `response` checkpoint 的 `valid` split 离线评估
- `serving/` 与 `api/` 已有第一版可调用接口；服务端合法动作生成、在线对局 session 和完整代理仍是后续工作。

## 当前可用命令

P800/Linux 上可以直接用仓库根目录的 [start.sh](start.sh) 串起完整链路。默认读取 `/workspace/dataset`，其余产物仍写入当前仓库下的 `data/` 和 `outputs/`：

```bash
./start.sh
TASKS=turn ./start.sh
TASKS=response ./start.sh
CLEAN_DATA=1 RUN_CALIBRATION=1 ./start.sh
RUN_VALIDATE=0 ./start.sh
RUN_TRAIN=0 RUN_EVAL=0 ./start.sh
```

`TASKS` 控制训练和对应评估的任务，默认是 `turn response` 全跑；`RUN_VALIDATE=0` 只跳过独立 `validate-events` 报告，`build-dataset` 是否跳过内部回放校验仍由 `SKIP_BUILD_VALIDATION` 控制。
本轮 `round_wind` 会在 `preprocess` 阶段写入事件；如果要针对当前代码跑全量训练，建议使用 `CLEAN_DATA=1` 清掉仓库内旧的 `bronze_events`、`silver_samples` 和 manifest 后从头重建。`RUN_CALIBRATION=1` 会在训练后对 response checkpoint 扫描 `pass_bias` 与 `no_yaku_call_penalty`。

```bash
python -m mjai.cli preprocess --input <raw_xml_dir> --output <bronze_events_dir>
python -m mjai.cli validate-events --input <bronze_events_dir> --report <report_json>
python -m mjai.cli build-dataset --input <bronze_events_dir> --output <samples_dir> --workers 8
python -m mjai.cli make-manifest --input <samples_dir> --output <manifest_dir>
python -m mjai.cli train --config <train_config_yaml>
python -m mjai.cli eval --ckpt <checkpoint_path> --split valid --report <eval_report_json>
python -m mjai.cli calibrate-response --ckpt <response_checkpoint_path> --split valid --manifest <manifest_jsonl> --report <calibration_report_json> --pass-biases 0,0.16,0.3,0.4 --no-yaku-call-penalties 0,0.2,0.4
python -m mjai.cli sample-response-errors --ckpt <response_checkpoint_path> --split valid --manifest <manifest_jsonl> --output <response_errors_jsonl> --targets "pass->chi,pass->pon,chi->pass,pon->pass" --pass-bias 0.3 --no-yaku-call-penalty 0.4
python -m mjai.cli sample-turn-errors --ckpt <turn_checkpoint_path> --split valid --manifest <manifest_jsonl> --output <turn_errors_jsonl>
python -m mjai.cli inspect-sample --dataset <samples_dir> --index 100
python -m mjai.cli inspect-sample --dataset <samples_dir> --sample-id <sample_id>
python -m mjai.cli dump-case --game-id <game_id> --kyoku-id <kyoku_index> --step-id <step_index>
python -m mjai.cli serve-api --config configs/api_local.yaml
pytest
```

说明：

- `preprocess` 会把天凤 XML 转成事件分片 Parquet，并输出 `preprocess_summary.json`。
- `validate-events` 会回放校验事件分片，并输出包含事件分布、失败原因和回放摘要的 JSON 报告。
- `build-dataset` 当前从 `bronze_events` 构建 `turn/response` Parquet 样本，并输出标签分布摘要；`--workers N` 会按事件分片并行构建，Windows 本机建议先从 `6` 或 `8` 起步；当前真实夹具的 `turn` 标签已覆盖 `discard / riichi_discard / tsumo / ankan / kyuushu_kyuuhai`，构建链路本身已支持 `kakan`。
- `make-manifest` 会按 `game_id` 生成 `train/valid/test` 清单，并输出 split 样本统计与 warning。
- `train` 当前支持最小 `turn/response` baseline，可通过 `train.turn_class_weights` / `train.response_class_weights` 做样本级类别加权，并输出 `config.resolved.yaml`、`train_metrics.json`、`best.pt`、`final.pt`。
- `eval` 当前支持最小离线评估，输出 `loss`、`top1/top3/top5 accuracy`、`macro top-k`、按动作类别聚合指标、主类/非主类分组、候选规模分桶和错例导出，并生成 JSON 报告。
- `calibrate-response` 会固定 response checkpoint 扫描 pass/call score margin 和 `no_yaku_call_penalty`，为指定 false call 目标选择 non-pass F1 最好的推理 operating point，并输出 `no_yaku_false_call_rate` / `no_yaku_false_call_share` 等无役鸣牌诊断指标。
- `sample-response-errors` 会重新跑 response manifest 推理，并按 `label->pred` confusion 分桶做 reservoir sampling；`--pass-bias` 与 `--no-yaku-call-penalty` 会共同决定抽样口径。
- `sample-turn-errors` 会重新跑 turn manifest 推理，并按 label kind 与 label rank 分桶做 reservoir sampling；默认抽 `discard` 的 `rank=2/3` 样本。
- `inspect-sample` 当前支持从样本目录或单个 parquet 文件按索引或 `sample_id` 抽查样本。
- `dump-case` 当前支持从 `bronze_events` 中定位目标决策点的原始事件上下文。
- `serve-api` 当前启动第一版 FastAPI 推理服务，要求配置中的 turn/response checkpoint 路径在本机存在，且调用方在预测请求中传入 `legal_actions`。
- `replay-eval`、`infer-on-state` 仍未实装。

训练性能说明：

- `build-dataset` 会在写出 parquet 时离线预编码 `state_features_bin` 和 `candidate_features_bin`，包括 `turn` discard 的向听数、有效牌、牌河时序等候选特征；训练和评估阶段只做 float32 binary 到 tensor 的拷贝，不再在每个 epoch 的 DataLoader worker 中重复计算这些规则特征。
- 这次样本 schema 已更新，且 2026-06-01 起 `preprocess` 会从 Tenhou `INIT seed` 写入 `round_wind`，用于修正 response 场风役牌路线和无役风险判断；旧的 `data/bronze_events` / `data/silver_samples` 不能继续用于训练，变更后需要重新执行 `preprocess -> build-dataset -> make-manifest`。
- Windows 本地 smoke 配置默认使用 `train.num_workers: 0`，避免 spawn worker 复制整份内存数据集；Linux/P800 可以继续通过配置提高 `num_workers`，但应以训练日志里的 `data_wait` 和 `samples_per_s` 判断是否真的收益。
- 训练过程会按 `train.log_every_n_steps` 输出 batch 进度日志，包含 `batch/total`、`batch_time`、`data_wait`、`compute_time`、窗口平均耗时、samples/s、窗口 loss/accuracy 和 ETA；其中 `data_wait` 用于定位 DataLoader 或 batch collation 是否仍是瓶颈。

## 当前缺口

- 规则引擎还不是完整日麻规则实现；虽然已补上自家舍牌振听、同巡振听、立直后振听、`kyuushu_kyuuhai` 和第一版 `chankan`，但仍缺少更完整的和牌机会放弃语义、其他途中流局边界与完整计分结算等能力。
- 诊断输出已经有第一版，并已打通 `eval -> inspect-sample -> dump-case` 的轻量跳转链路，但还缺少更细的错例聚合、固定 case 集合和更丰富的数据质量报告。
- 当前训练与评估已覆盖 `turn/response` 单任务 baseline 与多任务对照 run；类别权重仍是手工配置，自动权重估计、采样均衡、系统化错误分析工具和服务端合法动作生成仍待推进。

## 文档索引

- [CHANGELOG.md](CHANGELOG.md)：版本摘要。
- [docs/README.md](docs/README.md)：文档总索引和推荐阅读顺序。
- [docs/releases/v1.1.0.md](docs/releases/v1.1.0.md)：v1.1.0 发布说明。
- [docs/status.md](docs/status.md)：当前实现进度、已验证结果、主要缺口和建议的下一步任务。
- [docs/smoke-training.md](docs/smoke-training.md)：本地 smoke 数据、训练、评估和调试工作流。
- [docs/rule-coverage.md](docs/rule-coverage.md)：当前规则引擎覆盖范围与未实现边界。
- [docs/architecture-plan.md](docs/architecture-plan.md)：总体 AI、数据、模型和训练路线。
- [docs/engineering-guide.md](docs/engineering-guide.md)：工程设计与目录、CLI、配置约定。

## 建议的下一步

1. 用 `sample-turn-errors` 针对 40w `best_turn.pt` 抽 `ankan/kakan`、`riichi_discard` 和 `discard@rank2/3`，确认主候选的少数类回退边界。
2. 对比 05-10 improved `pass_bias=0.3` 与 40w `best_response.pt + pass_bias=0.0 + no_yaku_call_penalty=0.4` 的人工/在线 response case。
3. 继续 DDP 吞吐、卡数和稳定性调优，避免把工程加速变量和模型质量变量混在同一轮实验里。
