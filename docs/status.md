# 项目进度状态

更新时间：`2026-07-31`

## 1. 总体判断

当前项目已经明显超过“纯规划阶段”，可以视为：

- `Phase 0` 第一版已完成：规则与数据的最小闭环已经跑通。
- `Phase 1` 已推进到大数据单任务 baseline、第一轮特征复训、多任务调度实验和 40w DDP 训练之后：`turn/response` 都已经有 P800 baseline、improved、v1.1/v1.2 multitask、v1.2 safety 单任务和 40w multitask DDP 的 valid/test eval 与 response calibration 对照。

换句话说，仓库当前已经从“只能产样本”进入“能用大数据训练、归档 run、做阈值校准、分析模型取舍，并提供第一版 HTTP 推理接口”的阶段。v1.0.0 已固定为 2026-05-10 P800 improved 单任务 baseline；v1.1 多任务结果显示 `turn` 明确受益，但早期 `response` 经 raw eval 与 valid/test calibration 复核后仍不能替代 improved 单任务。v1.2 `batch_interleave` 修复了部分 response macro/`daiminkan` 问题，但未解决核心 false/missed/F1/少数类取舍。2026-06-04 单任务 v1.2 safety/features 复训把 20w `turn` 推到 `top1=71.98%`、`macro_top1=91.53%`、`non_discard=91.99%`；`response` raw false call 降到 `1.009%`、non-pass F1 升到 `73.24%`，但 exact `chi/daiminkan` 低于 05-10 improved。2026-06-24 40w v1.1 DDP 多任务训练进一步刷新当前 turn 主候选：`best_turn.pt` test `top1=72.88%`、`top3=96.04%`、`top5=99.08%`、`riichi_discard=89.13%`、`non_discard=92.68%`；少数回退点是 `macro_top1` 略低于 06-04，`ankan/kakan` 需要错例复核。40w `best_response.pt` raw false call 降到 `0.987%`、non-pass F1 升到 `73.64%`、multi-candidate top1 升到 `87.87%`，但 missed call 为 `8.44%`，exact `chi/pon/daiminkan` 仍不如 05-10 improved。当前 response 默认 serving 配置已按“优先压低错误鸣牌”的产品口径切到 40w `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8`；05-10 improved `pass_bias=0.3` 与 40w `pass_bias=0.0 + no_yaku_call_penalty=0.4` 继续作为对照口径保留。DDP 已在 P800 40w 训练、eval 和 calibration 中跑通；服务端合法动作生成和在线对局代理仍未完成。

2026-07-31 代码审计还确认了以下实现边界：

- 当前模型是 MLP candidate ranker，不是配置名所暗示的 Transformer；value head/value 训练尚未实现。
- `replay-eval` 与 `infer-on-state` 只是已注册的占位命令，并会打印 pending 后返回 `0`。
- API 是“调用方提供候选后的排序服务”；它做结构与基础取值校验，不会根据 state 重新证明候选动作合法。
- Git 已有 `v1.2.0` 标签，但 `pyproject.toml`、`mjai.__version__` 和 OpenAPI 仍为 `1.1.0`，这是当前已知的版本元数据不一致。
- PyTorch 必须按本机设备环境另行安装，当前不由 `pyproject.toml` 依赖自动提供。
- `configs/multitask_p800.yaml` 的输出目录带 `_ddp`，但 `start.sh` / compose 的默认 `MULTITASK_EVAL_CKPT` 仍指向旧的不带 `_ddp` 路径；在脚本默认值修正前必须显式覆盖。

## 2. 已验证结果

以下结果已经在仓库内实际验证：

- 当前测试树经 `python -m pytest --collect-only -q` 可收集 `134` 个 case。全量测试包含多次完整 fixture 数据构建与 smoke train，本轮没有在文档维护时限内取得完整结束结果，因此不把收集数写成 passed 数。
- 本轮针对规则、特征、配置、指标、校准、serving predictor/API 路由运行了两组定向回归，共 `86 passed`；`python -m compileall -q mjai tests`、`git diff --check` 和文档相对链接检查通过。
- 最近一次文档记录的完整回归为 `127 passed`（2026-06-25）；本轮将其保留为历史验证，不再称为当前用例总数。
- `python -m mjai.cli preprocess --input tests/fixtures/tenhou --output <tmp>` 可运行。
- 2026-07-31 对 `tests/fixtures/tenhou` 中当前 `21` 个 XML 夹具重新实测，生成 `7386` 条事件并写入 `bronze_events`。
- `python -m mjai.cli validate-events --input <tmp> --report <tmp>` 可运行。
- 同一批事件分片实测回放结果为：
  - `files=21`
  - `valid=21`
  - `failed=0`
  - `total_events=7386`
  - 可输出事件分布、失败原因聚合、总局数/弃牌数/副露数摘要
- `python -m mjai.cli build-dataset --input <tmp> --output <tmp>` 可运行，且支持 `--workers N` 按事件分片并行构建。
- `python -m mjai.cli build-dataset --input <tmp> --output <tmp> --workers 2` 已在 Windows fixture smoke 中验证通过。
- 对当前 `21` 个 XML 夹具实测，生成：
  - `3532` 条 `turn` 样本
  - `10506` 条 `response` 样本
  - 可输出 `turn/response` 标签动作分布；当前真实夹具的 `turn` 已覆盖 `discard / riichi_discard / tsumo / ankan / kakan / kyuushu_kyuuhai`
- `python -m mjai.cli make-manifest --input <tmp> --output <tmp>` 可运行。
- 同一批夹具可生成 manifest，实测分桶结果为：
  - `train_games=19`
  - `valid_games=1`
  - `test_games=1`
- `python -m mjai.cli train --config <tmp>` 可运行。
- 2026-05-20 训练引擎已接入 PyTorch DDP：`train.distributed: true` 需要通过 `torchrun` / `python -m torch.distributed.run` 启动；训练集 streaming dataset 按 rank 切分 shard；验证、checkpoint 和指标写盘只在 rank 0 执行；DDP checkpoint 会保存未带 `module.` 前缀的模型权重。该能力已通过本地编译、配置、dataset sharding 和 multitask smoke 回归，并在 2026-06-24 P800 40w `multitask_p800_v1_1_ddp` 训练、eval 和 calibration 中完成实机验证；后续重点转为吞吐、卡数和稳定性调优。
- 2026-06-26 起，offline eval 与 response calibration 也支持 `eval.distributed: true` 的多 rank 分片、指标归并和 rank 0 报告写出；默认配置仍为 `false`，启用时必须通过 `torchrun` 启动且 `WORLD_SIZE > 1`。
- 以单局清单裁剪出的最小训练集实测，`turn` 与 `response` 任务都可完成 `1` 个 epoch 的 smoke train，并落盘：
  - `config.resolved.yaml`
  - `train_metrics.json`
  - `best.pt`
  - `best_accuracy.pt`
  - `best_loss.pt`
  - `best_macro.pt`
  - `final.pt`
- `python -m mjai.cli eval --ckpt <tmp> --split valid --report <tmp>` 可运行。
- 离线 `eval` 现在默认使用 streaming dataset：启动时只读取 manifest 与 shard 元信息，迭代时按 parquet shard 读取预编码特征和错例报告需要的动作 JSON，不再把整份 valid/test split 常驻内存。`eval.num_workers` 可单独配置；未配置时沿用 `train.valid_num_workers`。
- 对最小 `turn` 与 `response` smoke checkpoint 实测，可输出：
  - `loss`
  - `top1 / top3 / top5 accuracy`
  - 按动作类别聚合的样本数与 `top-k` 指标
  - 按候选动作数量分桶的 `top-k` 指标
  - 首批错例样本的 label、top-k 预测、legal actions，以及可直接执行的 `inspect-sample` / `dump-case` 跳转命令
  - JSON 评估报告
- response 评估报告已补充 response 专用诊断：`single_candidate` / `multi_candidate`、`pass` vs `non_pass` precision / recall / F1、`false_call_rate`、`missed_call_rate`、候选集合聚合与 `label->pred` confusion。
- 2026-06-01 已修正 response 役牌路线特征的场风来源：Tenhou `INIT seed` 会解析出 `round_wind`，样本状态和服务端请求校验都会保留该字段，避免南场/西场默认按东场判断役牌路线。
- `python -m mjai.cli calibrate-response --ckpt <response_ckpt> --split valid --manifest <manifest> --report <report>` 可运行；它会扫描 pass/call score margin 和 `no_yaku_call_penalty`，并为 `false_call_rate <= 0.010 / 0.008 / 0.005` 等目标选择 `non_pass_f1` 最好的 operating point。报告已包含 `no_yaku_false_call_rate` / `no_yaku_false_call_share`。
- `python -m mjai.cli sample-response-errors --ckpt <response_ckpt> --split valid --manifest <manifest> --output <jsonl>` 可运行；它会按 `label->pred` confusion 分桶做 reservoir sampling，默认目标为 `pass->chi / pass->pon / chi->pass / pon->pass`，`--pass-bias` 与 `--no-yaku-call-penalty` 会共同决定抽样口径，`--seed` 默认继承 checkpoint 全局 `seed`。
- `python -m mjai.cli sample-turn-errors --ckpt <turn_ckpt> --split valid --manifest <manifest> --output <jsonl>` 可运行；它会按 label kind 与 label rank 分桶做 reservoir sampling，默认抽 `discard@rank2` 与 `discard@rank3`，用于 turn top-k 错例分析。
- 2026-05-03 response action feature 增强后复训已完成并记录在 [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md)：训练过滤后保留 `141423 / 1490034` 条 response 训练样本；最新 `best.pt` 在 valid/test 上分别达到 `multi_candidate top1=0.8521/0.8502`、`multi_candidate macro top1=0.7882/0.8697`、`non_pass_f1=0.7215/0.7189`、`false_call_rate=0.0111/0.0111`、`missed_call_rate=0.1425/0.1466`。
- 2026-05-06 P800 大数据单任务 baseline 已完成，训练产物已归档到 `runs/2026-05-06_p800_baseline/`，分析记录见 [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md) 第 10 节。
  - `turn_p800/best.pt`：train `83077288` 条、valid `10404270` 条、test `10393095` 条；test `top1=0.7008`、`top3=0.9446`、`top5=0.9853`、`macro_top1=0.9134`，其中 `discard top1=0.6958`、`riichi_discard top1=0.8534`。
  - `response_p800/best.pt`：原始 train `247177362` 条，过滤 pass-only 后有效 train `23120538` 条；test `multi_candidate top1=0.8619`、`multi_candidate macro top1=0.8742`、`non_pass_f1=0.7131`、`false_call_rate=0.0118`、`missed_call_rate=0.0644`、`chi top1=0.8376`、`pon top1=0.9528`。
  - 本轮 P800 数据版本已归档到 `runs/_datasets/tenhou_20w_2026-05-06/`：`preprocess_summary.json` 记录 `212163` 个 game、`215224122` 条事件；`event_validation_20w.json` 记录 `files_seen=212163`、`files_valid=212163`、`files_failed=0`、`error_counts={}`；`build_summary.json` 记录 `103874653` 条 turn、`309055971` 条 response；`manifest_summary.json` 与 `train/valid/test.jsonl` 逐行累加结果完全一致，并和 P800 训练日志样本数对齐。
- 2026-05-10 P800 improved 复训已完成，训练产物已归档到 `runs/2026-05-10_p800_improved/`，分析记录见 [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md) 第 11 节。
  - `turn` 明确进步：test `top1=0.7075`、`top3=0.9493`、`top5=0.9868`，相对 baseline 分别提升 `+0.67pp / +0.48pp / +0.15pp`；`discard top1=0.7024` 提升 `+0.66pp`，`riichi_discard top1=0.8713` 提升 `+1.78pp`。副作用是 `ankan top1` 从 `0.9602` 降到 `0.8988`，需要错例排查。
  - `response` 是更保守的候选：test `multi_candidate top1=0.8686`、`multi_candidate macro top1=0.8843`、`non_pass_f1=0.7214`、`false_call_rate=0.0111`，均优于 baseline；但 `missed_call_rate` 从 `0.0644` 升到 `0.0728`，`chi top1` 从 `0.8376` 降到 `0.8198`，`pon top1` 从 `0.9528` 降到 `0.9475`。
  - response calibration 已完成 valid/test。improved 在同一 `pass_bias` 下误叫更少但漏叫更多；baseline 的默认候选 `pass_bias=0.4` 不直接迁移到 improved。v1.0.0 默认 response operating point 定为 `pass_bias=0.3`：test `false_call_rate=0.925%`、`missed_call_rate=9.29%`、`non_pass_f1=74.18%`、`multi_candidate top1=88.40%`、`chi/pon top1=78.4%/93.1%`。
- 2026-05-12 P800 multitask v1.1 首轮训练已完成，训练产物已归档到 `runs/2026-05-12_p800_multitask_v1_1/`，分析记录见 [multitask-training.md](multitask-training.md) 第 7 节和 run 目录下的 `analysis.md`。
  - `turn` 明确优于 2026-05-10 improved：test `top1=0.7169`、`top3=0.9538`、`top5=0.9884`，相对 improved 分别提升 `+0.94pp / +0.45pp / +0.16pp`；`discard top1=0.7118`、`riichi_discard top1=0.8803`、`non_discard top1=0.9207` 均提升，`ankan top1` 也从 improved 的 `0.8988` 恢复到 `0.9186`。
  - `response` 仍不能替代 improved 单任务：补跑 `best_response.pt` 后，raw test `multi_candidate top1=0.8646`、`multi_candidate macro top1=0.8518`、`non_pass_f1=0.7149`、`false_call_rate=0.0114`，比 multitask `best.pt` 更接近 improved，但仍弱于 improved 的 `0.8686 / 0.8843 / 0.7214 / 0.0111`。valid/test calibration 也未反转结论：test `pass_bias=0.3` 时 multitask best_response 为 `false_call=0.933%`、`missed=9.91%`、`non_pass_f1=73.73%`，弱于 improved 的 `0.925% / 9.29% / 74.18%`；test `pass_bias=0.4` 时 false call 基本持平，但 missed call 更高、F1 更低，且 `daiminkan` 明显偏低。不应把 multitask response 作为默认候选。
- 2026-05-13 P800 multitask v1.2 interleave 训练已完成，训练产物已归档到 `runs/2026-05-13_p800_multitask_v1_2_interleave/`，分析记录见 [multitask-training.md](multitask-training.md) 第 8 节和 run 目录下的 `analysis.md`。
  - `turn best.pt` 达到当前最高 overall top1：test `top1=0.7180`、`top3=0.9539`、`top5=0.9885`。补跑 `best_turn.pt` 后，少数类明显恢复：`top1=0.7173`、`macro=0.9125`、`riichi_discard=0.8701`、`non_discard=0.9143`、`ankan=0.9231`。但 v1.1 仍有更高 macro、`riichi_discard`、`kakan` 和 `non_discard`，因此 v1.2 `best_turn.pt` 是候选，不是明确替代。
  - `response best_response.pt` 与 `best.pt` raw test 完全一致：`macro_top1=0.9017`、`multi_candidate top1=0.8604`、`multi_candidate macro=0.8780`、`non_pass_f1=0.7117`、`false_call_rate=1.203%`、`missed_call_rate=6.07%`、`daiminkan=0.6232`。相对 v1.1 `best.pt` 更健康，但仍弱于 v1.0.0 improved 的核心 response 口径。valid/test calibration 未反转结论：test `pass_bias=0.4` 时 v1.2 为 `false_call=0.969%`、`missed=8.39%`、`F1=73.87%`、`daiminkan=47.2%`，仍弱于 improved 默认 `pass_bias=0.3` 的 `0.925% / 9.29% / 74.18% / 61.6%`。response 默认候选不变。
  - 默认配置决策：`configs/multitask_p800.yaml`、`start.sh` 和 `docker-compose.p800.yml` 已回到 v1.1 路径。v1.2 复现配置已收拢到 `configs/experiments/multitask_p800_v1_2_interleave.yaml`，不再放在主配置入口目录。
- 2026-05-20 基于 commit `7a7ad21` 的 P800 multitask v1.1 复训已完成，训练产物已归档到 `runs/2026-05-20_p800_multitask_v1_1_7a7ad21/`，分析记录见 [multitask-training.md](multitask-training.md) 第 9 节和 run 目录下的 `analysis.md`。
  - `turn best_turn.pt` 是当前 20w run 的 overall 高点：test `top1=0.7188`、`top3=0.9559`、`top5=0.9892`、`discard=0.7142`。但 `macro=0.9106`、`riichi_discard=0.8594`、`non_discard=0.9073` 均弱于 old v1.1，说明该 checkpoint 更偏 overall/discard，不是对 old v1.1 的均衡替代。
  - `response best_response.pt` raw test 为 `multi_candidate top1=0.8641`、`multi_candidate macro=0.8760`、`non_pass_f1=0.7141`、`false_call_rate=1.142%`、`missed_call_rate=7.53%`、`daiminkan=0.6459`。相对 old v1.1 `best_response.pt` 明显恢复 `daiminkan`，但仍弱于 improved 单任务的 `multi_candidate top1=0.8686`、`non_pass_f1=0.7214`、`false_call_rate=1.105%` 和 `daiminkan=0.6817`。
  - valid/test calibration 已归档。test `pass_bias=0.3` 下，`7a7ad21 best_response.pt` 为 `false_call=0.956%`、`missed=9.55%`、`F1=73.51%`、`multi top1=88.02%`、`daiminkan=56.7%`，弱于 improved 默认 `0.925% / 9.29% / 74.18% / 88.40% / 61.6%`；test `pass_bias=0.4` 下也仍弱于 improved 的同 bias 结果。因此 response 默认候选不变。
- 2026-06-04 基于 `start.sh` 的 P800 单任务复训已完成，训练产物已归档到 `runs/2026-06-04_p800_single_v1_2_safety/`，可提交分析见 [training-analysis-2026-06-04.md](training-analysis-2026-06-04.md)，run 目录下也有 `analysis.md` 与 `run_summary.json`。
  - `turn_p800/best.pt` 是当时最强 turn 候选：test `top1=71.98%`、`top3=95.72%`、`top5=98.96%`、`macro_top1=91.53%`、`discard=71.49%`、`riichi_discard=87.89%`、`non_discard=91.99%`。相对 05-20 multitask `best_turn.pt`，overall 和少数类稳定性均更好；随后已被 06-24 40w `best_turn.pt` 的主指标刷新。
  - `response_p800/best.pt` raw test 为 `multi_candidate top1=87.67%`、`non_pass_f1=73.24%`、`false_call_rate=1.009%`、`missed_call_rate=8.28%`、`chi=79.68%`、`pon=94.08%`、`daiminkan=61.86%`。它比 05-10 improved 更会压误叫，但 exact `chi` 与 `daiminkan` 更弱，不应直接无条件替换默认 response。
  - response calibration 显示 `best.pt` 与 `best_response.pt` 曲线一致。当前建议新增实战候选为 `pass_bias=0.0 + no_yaku_call_penalty=0.4`：test `false_call=0.915%`、`no_yaku_false=0.242%`、`missed=9.49%`、`F1=74.27%`、`multi top1=88.45%`、`chi/pon/daiminkan=77.0%/93.4%/61.5%`。05-10 improved `pass_bias=0.3` 仍保留为稳定 serving baseline。
- 2026-06-24 P800 multitask v1.1 DDP 40w 训练已完成，训练产物已归档到 `runs/2026-06-24_p800_multitask_v1_1_ddp_40w/`，数据集已归档到 `runs/_datasets/tenhou_40w_2026-06-23/`，分析记录见 [multitask-training.md](multitask-training.md) 第 10 节和 run 目录下的 `analysis.md`。
  - 40w 数据集质量证据完整：`379742` 局、`385081321` 条事件，`validate-events` 为 `files_seen=379742`、`files_valid=379742`、`files_failed=0`、`error_counts={}`；构建出 `185846893` 条 turn 与 `552934725` 条 response 样本，manifest test split 为 `18681274` 条 turn 和 `55580670` 条 response。
  - `multitask/best_turn.pt` 是当前最强 turn 候选：test `top1=72.88%`、`top3=96.04%`、`top5=99.08%`、`macro_top1=91.35%`、`discard=72.40%`、`riichi_discard=89.13%`、`non_discard=92.68%`。相对 06-04 single，overall、discard、riichi_discard 和 non_discard 均提升；`macro_top1`、`ankan`、`kakan` 轻微或中等回退，需要错例复核。
  - `multitask/best_response.pt` raw test 为 `multi_candidate top1=87.87%`、`non_pass_f1=73.64%`、`false_call_rate=0.987%`、`missed_call_rate=8.44%`、`chi=79.42%`、`pon=93.86%`、`daiminkan=62.31%`。它是当前 raw binary/multi 口径最强 response 候选，但相对 05-10 improved 仍牺牲 exact `chi/pon/daiminkan` 和 missed call。
  - response calibration 中，当前最值得人工复核的 40w 候选为 `best_response.pt` + `pass_bias=0.0 + no_yaku_call_penalty=0.4`：test `false_call=0.884%`、`no_yaku_false=0.217%`、`missed=9.85%`、`F1=74.72%`、`multi top1=88.70%`、`chi/pon/daiminkan=76.3%/93.1%/61.9%`。自动选出的 `pass_bias=1.0 + no_yaku_call_penalty=0.4` 虽有更高 F1，但 missed call 达 `18.54%`、`daiminkan=28.5%`，不适合作默认。
  - 2026-06-25 对正式 valid split 追加扩展扫描，当时报告路径为 `outputs/reports/formal_calibration_low_false_call_valid_20260625_151255.json`，覆盖 `55,352,691` 条 response 样本。当前工作区未保留这份原始 JSON，以下数值可在对应 run 的 `analysis.md` 追溯，但证据归档仍不完整。若目标明确偏向“错过鸣牌没关系，但尽量压低错误鸣牌”，推荐极保守档 `pass_bias=2.0 + no_yaku_call_penalty=0.8`：valid `false_call=0.1972%`、`no_yaku_false=0.0083%`、`missed=31.52%`、`precision=85.73%`、`recall=68.48%`、`F1=76.14%`。代表性参数对比如下：

| pass_bias | no_yaku_penalty | false call | missed call | precision | recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2.0` | `1.2` | `0.1928%` | `32.31%` | `85.87%` | `67.69%` | `75.70%` |
| `2.0` | `0.8` | `0.1972%` | `31.52%` | `85.73%` | `68.48%` | `76.14%` |
| `2.0` | `0.2` | `0.2115%` | `30.01%` | `85.13%` | `69.99%` | `76.82%` |
| `1.8` | `0.8` | `0.2321%` | `28.91%` | `84.13%` | `71.09%` | `77.06%` |
| `1.8` | `0.4` | `0.2428%` | `27.87%` | `83.72%` | `72.13%` | `77.49%` |
| `1.6` | `0.4` | `0.2853%` | `25.35%` | `81.91%` | `74.65%` | `78.11%` |
| `1.6` | `0.2` | `0.2950%` | `24.72%` | `81.54%` | `75.28%` | `78.29%` |
| `1.4` | `0.4` | `0.3332%` | `22.94%` | `80.01%` | `77.06%` | `78.51%` |
| `1.4` | `0.2` | `0.3454%` | `22.27%` | `79.57%` | `77.73%` | `78.64%` |
| `1.2` | `0.4` | `0.3878%` | `20.67%` | `77.97%` | `79.33%` | `78.64%` |
| `1.2` | `0.2` | `0.4035%` | `19.96%` | `77.44%` | `80.04%` | `78.72%` |
| `1.0` | `0.4` | `0.4501%` | `18.52%` | `75.80%` | `81.48%` | `78.54%` |
| `1.0` | `0.2` | `0.4693%` | `17.77%` | `75.20%` | `82.23%` | `78.56%` |
- `python -m mjai.cli inspect-sample --dataset <tmp> --index <n>` 可运行。
- `python -m mjai.cli inspect-sample --dataset <tmp> --sample-id <id>` 可运行。
- 对生成的 `silver_samples` 实测，可按全局索引输出：
  - `sample_id / game_id / kyoku_index / step_index / actor`
  - label action
  - legal actions
  - 状态快照
  - 来源 parquet 文件与文件内行号
- `python -m mjai.cli dump-case --game-id <id> --kyoku-id <id> --step-id <id>` 可运行。
- 对生成的 `bronze_events` 实测，可按 `game_id / kyoku_index / step_index` 输出：
  - 目标事件
  - 同一局内的完整事件列表
  - 目标事件前后上下文
  - 来源 parquet 分片路径
- 真实牌谱扩大批次曾在当前 `data/` 目录下完成一次端到端数据主线验证，命令链路为 `preprocess -> validate-events -> build-dataset --workers N -> make-manifest`。这批 2026-05 结果是历史本地质量基线；它早于 `round_wind` 和后续 turn 特征修正，与当前 schema/语义不兼容，不能直接作为当前 checkpoint 的训练数据。
  - `preprocess_summary.json`：`4233` 局，`1322990` 条事件，无 warning。
  - `validate.json`：`files_seen=4233`，`files_valid=4233`，`files_failed=0`，事件校验失败率 `0.00%`；总局数 `15275`，总弃牌 `625144`，总副露 `29935`。
  - `build_summary.json`：`631016` 条 `turn` 样本，`1876854` 条 `response` 样本；`games_with_zero_turn_samples=0`，`games_with_zero_response_samples=0`，无 warning。
  - 非法标签率复核：逐行比对 `label_action_json` 是否存在于 `legal_actions_json` 后，`turn` 为 `0 / 631016 = 0.00%`，`response` 为 `0 / 1876854 = 0.00%`。
  - `turn` 标签分布：`discard=614528` (`97.3871%`)，`riichi_discard=10616` (`1.6824%`)，`tsumo=4636` (`0.7347%`)，`ankan=669` (`0.1060%`)，`kakan=474` (`0.0751%`)，`kyuushu_kyuuhai=93` (`0.0147%`)。
  - `response` 标签分布：`pass=1838052` (`97.9326%`)，`pon=17489` (`0.9318%`)，`chi=11203` (`0.5969%`)，`ron=10003` (`0.5330%`)，`daiminkan=100` (`0.0053%`)，`chankan=7` (`0.0004%`)。
  - split 规模：`train=3362` 局、`500969` 条 `turn`、`1490034` 条 `response`；`valid=426` 局、`64229` 条 `turn`、`191082` 条 `response`；`test=445` 局、`65818` 条 `turn`、`195738` 条 `response`。
  - 结论：数据构建质量可以进入训练 smoke/小规模正式训练，但类别极度不均衡已经成为训练主线的主要风险之一，整体 accuracy 不能单独作为有效模型质量判断。

## 3. 按模块看当前状态

### 已完成或基本完成

- `mjai/data/tenhou/`
  - 天凤 XML 流式解析已实现。
  - 副露解码已实现，覆盖 `chi / pon / daiminkan / ankan / kakan`；`daiminkan` 已按 Tenhou anchor tile 判定被鸣牌，避免赤牌明杠误消耗手牌。
- `mjai/rules/`
  - 手牌状态、回放状态、弃牌/吃碰杠/荣和的第一版合法动作生成已实现。
  - `turn` 合法动作已支持 `discard / riichi_discard / tsumo / ankan / kakan / kyuushu_kyuuhai`。
  - `response` 的 `ron` 已补上自家舍牌振听、同巡振听与立直后振听过滤。
  - `response` 已补上第一版 `chankan`，且 `kakan` 响应窗口不再暴露 `chi / pon / daiminkan`。
  - 基础胡牌判断与立直前听牌检查已实现。
  - 回放校验器已实现，可验证弃牌合法性和副露消耗牌一致性。
- `mjai/data/builders/`
  - `bronze_events` 预处理器已实现。
  - 事件分片回放校验器已实现。
  - `turn` 与 `response` 样本构建器已实现，当前真实夹具已产出 `tsumo`、`ankan` 与 `kyuushu_kyuuhai` 的 `turn` 样本，`kakan` 与 `chankan` 也已进入合法动作与样本构建链路。
  - `build-dataset` 已支持 `--workers N` 并行处理事件分片，优先使用进程池；当 Windows 环境触发进程池限制时会回退到线程池。
  - Parquet 写出与 manifest 生成已实现。
  - 事件分布、动作分布、split 样本统计和 warning 已实现第一版。
- `mjai/data/datasets/`
  - `build-dataset` 写出的 parquet 已包含 `state_features_bin`、`candidate_features_bin`、`label_index` 和 `label_kind`；训练和评估阶段不再逐 epoch 重新编码状态/动作特征。
  - `turn` discard 的向听数、有效牌、形状摘要等候选特征已经移到样本构建阶段离线计算，避免 DataLoader worker 在训练期反复执行 CPU 密集的规则搜索。
  - 默认训练/eval 主线使用 streaming iterable dataset：初始化只读取 manifest 与 shard 元信息，迭代时按 shard/batch 读取预编码特征；eager 数据集只用于较小数据或调试。旧 schema 的 `silver_samples` 需要重新运行 `build-dataset` / `make-manifest` 后才能训练。若 parser 新增事件字段或样本状态字段，例如 `round_wind`，还必须先重新运行 `preprocess`。
- `configs/`
  - 数据、任务、模型、训练配置目录已建立。
- `mjai/features/`
  - 已有最小 `turn/response` 状态特征与动作特征编码。
  - 2026-05-19 已继续增强 turn / response 输入：turn state 新增四家立直状态、对手立直巡目和最多 192 个全局牌河顺序 token；turn discard 候选特征从 `85` 增至 `89`，新增按绝对座位拆开的现物信号，用于继续压低放炮并减少无役鸣牌。
  - response 候选动作特征已补充第一版 call 收益/代价信息：叫牌前后向听、有效牌数量、有效牌枚数、赤牌/dora 消耗、对子/搭子消耗、役牌碰、chi 形状、巡目和剩余巡目。
  - 2026-05-09 已继续增强 response call/context 特征：`RESPONSE_ACTION_DIM` 从 `77` 增至 `91`，新增可见牌修正有效牌枚数、开副露后明显役路启发式、无役风险、七对子/国士等门清特殊路线破坏信号；样本状态同步写出 `all_melds`，用于扣除其他玩家副露里的可见牌。该特征是启发式，不等价于完整番种判定。
  - 2026-06-01 起，response 的役牌路线启发式会使用 `round_wind` 判断场风牌；该修正不改变特征维度，但会改变 response candidate 特征语义，下一轮训练数据需要重建。
- `mjai/models/`
  - 当前单任务和多任务模型都是 MLP candidate ranker；多任务仅共享 state MLP 主干，turn/response 保留各自输入投影、action encoder 和 head。当前没有 Transformer 或 value head。
- `mjai/training/`
  - 已有最小训练循环、`turn/response` 标签样本加权 loss、总体与宏平均 accuracy、按动作类别和主类/非主类分组的训练指标、checkpoint 落盘。
  - response 训练已支持 `train.response_min_train_candidate_count`，当前 response smoke / P800 配置默认用 `2` 过滤训练集中的 pass-only 单候选样本；valid/test 评估仍保留全量样本。
  - response 的 `best.pt` / `best_response.pt` 现在按 `non_pass_f1 - false_call_rate` 选择，避免只被 overall accuracy 或 pass-only 样本支配；`best_accuracy.pt`、`best_loss.pt`、`best_macro.pt` 仍保留。
  - 2026-05-03 action feature 增强后复训显示 response 已经达到进入更大数据 baseline 的门槛：false call 与 missed call 同时下降，`chi/pon` top1 也有明确提升。
  - 2026-05-06 P800 大数据 baseline 进一步确认当前配置有效：response 的 `chi/pon` exact action 明显提升，missed call 大幅下降；下一步重点转为 pass/call 校准和 hard negative 错例分析，而不是继续盲目调权重。
  - 已支持 `pin_memory`、`persistent_workers`、`prefetch_factor` 和 CUDA `non_blocking` 传输；Windows 本地 smoke 配置默认 `num_workers: 0`，避免 spawn worker 复制整份内存数据集。
  - 训练循环已接入 `train.precision` 的 CUDA/XPU autocast；P800 配置默认 `bf16`，CUDA 路径默认开启 TF32，并在启动日志中打印实际精度设置。
  - 训练指标聚合已减少 batch 内逐样本 `.item()` 同步：每个 batch 将 `correct`、per-sample loss、response rank/prediction 一次性搬到 CPU 后聚合，降低小 MLP ranker 训练时的 GPU 同步碎片。
  - response 训练 split 默认关闭完整 response 诊断，跳过 `legal_actions_json` 读取/解析和 label-rank/candidate-set 统计；valid/test 诊断仍保留，用于 `best_response.pt` 选择和离线评估。
  - train split 的按标签/主类分组指标已改为 `label_kind_ids` + `torch.bincount` 批量聚合，避免 `batch_size=8192/16384` 时 Python 逐样本循环拖慢训练。
  - 已支持由 `train.log_every_n_steps` 控制的 batch 进度日志，输出 batch 进度、`batch_time`、`data_wait`、`compute_time`、窗口平均耗时、samples/s、窗口 loss/accuracy 和 ETA。
- `mjai/eval/`
  - 已有第一版离线评估入口，可基于 checkpoint 复用训练配置做 split 级评估与 JSON 报告导出。
  - 已支持按动作类别、主类/非主类分组、候选规模分桶与首批错例导出。
  - response checkpoint 额外输出 `candidate_count > 1` 子集、pass/non-pass 二分类、false call / missed call 和候选集合维度指标；turn checkpoint 的 `response` 诊断字段为 `null`。
  - 已新增 response pass/call 校准扫描：固定 checkpoint，在一组 `pass_bias` 下输出 false call、missed call、non-pass precision/recall/F1、multi-candidate top1 和按动作类别 top1，并按 false call 目标挑选 operating point。
  - 已新增 response 错例抽样命令：重新跑 manifest 推理后按 `label->pred` confusion 抽样，避免误用 `eval.misclassified_examples` 里默认最多 `10` 条的调试样本。
  - 已新增 turn top-k 错例抽样命令：重新跑 manifest 推理后按 label kind 与 label rank 抽样，默认目标为 `discard@rank2` 与 `discard@rank3`。
- `mjai/serving/` 与 `mjai/api/`
  - 已完成第一版外部调用接口：`mjai/serving` 负责 checkpoint 加载、模型推理、候选排序和请求结构/基础取值校验；`mjai/api` 负责 FastAPI app factory、路由、Pydantic schema、API key middleware 和统一错误响应。
  - 已支持 `/health`、`/v1/models`、`/v1/predict/turn`、`/v1/predict/response`。
  - `response` 当前默认接入 06-24 40w `best_response.pt` 的极低错误鸣牌 operating point：`pass_bias=2.0` 与 `no_yaku_call_penalty=0.8`；配置中仍保留 `recall=0.0` 与 `conservative=0.16` 作为低 bias 对照档。
  - `response` 推理已支持 `no_yaku_call_penalty`，对无明显开副露役路的 `chi/pon/daiminkan` 候选在 calibrated 选择前额外扣分；`configs/api_local.yaml` 当前默认设为 `0.8`，用于进一步降低无役鸣牌。
  - 第一版仍要求调用方传入 `legal_actions`，服务端合法动作生成留到后续阶段。
- `tests/`
  - 规则、解析、样本构建、CLI、配置和数据写出都有测试覆盖。
  - `turn/response` smoke train、offline eval、response/turn 错例抽样、`inspect-sample`、`dump-case` 与 serving/API 路由都有回归测试。

### 部分完成

- `mjai/cli.py`
  - 子命令框架完整。
  - `preprocess`、`validate-events`、`build-dataset`、`make-manifest`、`train`、`eval`、`calibrate-response`、`sample-response-errors`、`sample-turn-errors`、`inspect-sample`、`dump-case` 已可用。
  - `replay-eval`、`infer-on-state` 仍是占位实现。
- 配置与训练参数
  - `configs/model/*.yaml` 中的 Transformer/value 字段，以及部分 `grad_accum_steps`、`val_every_n_steps`、`save_every_n_steps` 字段当前未被训练引擎消费；只有已在代码读取路径中的参数可视为生效配置。
  - `configs/task/value.yaml` 是预留文件，训练入口只接受 `turn/response/multitask`。
- 数据链路
  - 已经落地 `raw_xml -> bronze_events -> silver_samples` 两阶段流水线。
- 第一版诊断输出已经落地，并已支持从评估错例跳转到样本与事件上下文，但还缺少更细的失败样本聚合和固定 case 管理。

### 尚未开始或基本未开始

- `mjai/serving/` 的阶段 3 和阶段 4。

其中合法动作服务端生成与在线对局代理仍待后续实现；但纯推理内核和 FastAPI 薄封装已完成第一版。

## 4. 当前最关键的差距

### 4.1 数据流水线与基础诊断已经打通，但深入排查能力还偏弱

当前已经具备：

```text
raw_xml -> bronze_events -> validate-events -> silver_samples
```

并且已经有：

- 事件分布统计
- 错误类型聚合
- 动作标签分布
- split 样本数统计
- 小数据集空 split warning

当前已经足够进入更大规模训练试跑。后续数据诊断应围绕训练主线补强，优先输出：

- 大规模构建失败率与失败原因 Top-N
- `label_action not in legal_actions` 计数与样本定位
- 动作分布、候选规模分布和 split 规模

### 4.2 规则引擎仍是 first-pass 版本

当前规则层已经足够支撑“早期监督学习样本构建”，但还不足以称为完整日麻规则引擎。继续补规则边界的优先级应由训练数据和错例证据决定，主要候选缺口包括：

- 和牌只检查标准形/七对子/国士牌形，不检查役、番符和点数
- 立直后的摸切锁定、鸣牌限制和暗杠等待不变
- 更完整的和牌机会放弃语义与残余振听边界
- 多家荣和与响应仲裁
- 136 张守恒、摸弃顺序、手牌张数节奏、终局点数移动
- 更完整的流局处理

### 4.3 训练与评估已完成大规模验证，但高级能力仍有缺口

当前已经有：

- `turn` 状态特征编码
- `response` 状态特征编码
- `turn/response` 候选动作编码
- 最小共享编码器和 `turn/response` 候选排序头
- `turn/response` 训练循环
- `loss / accuracy / macro_accuracy`
- `turn/response` 离线 `top-k`、`macro_top-k` 评估与按类别聚合报告
- checkpoint 与训练指标落盘

类别不均衡处理现状：

- 当前代码已经对 `turn` 与 `response` 都做了第一版显式处理：`mjai/training/class_balance.py` 统一读取 `train.turn_class_weights` / `train.response_class_weights`，按 batch 内 `label_kinds` 构造 `sample_weights`，再传给 `policy_loss` 做样本级加权交叉熵。
- `configs/turn_local_smoke.yaml` 已配置 `riichi_discard / tsumo / ankan / kakan / kyuushu_kyuuhai` 高于 `discard` 的权重；`configs/response_local_smoke.yaml` 已配置 `chi / pon / daiminkan / ron / chankan` 高于 `pass` 的权重，并默认过滤 response 训练集中的 `candidate_count == 1` 样本。
- 训练指标现在会在 `train_metrics.json` 与 `epoch_metrics.jsonl` 中记录 `macro_accuracy`、`per_kind` 和 `class_groups`；其中 `class_groups` 会输出 `discard/non_discard` 或 `pass/non_pass`。response 还会记录 `multi_candidate`、`non_pass_f1`、`false_call_rate`、`missed_call_rate` 等专用诊断。离线 `eval` 报告也会输出 `macro_top1/top3/top5`、按动作类别 loss/top-k、主类/非主类分组和候选规模分桶。
- 仍未解决的部分：权重仍是手工常量，不会根据 manifest 或 `build_summary.json` 自动估计；训练 DataLoader 仍是普通 shuffle，没有 `WeightedRandomSampler` 或分层采样；下一轮训练需要根据 `non_discard`、`non_pass`、`ron / chi / pon / riichi_discard / tsumo` 等指标决定是否调权重或引入采样均衡。

多头边界现状：

- v1.1 已新增 `MultitaskMahjongPolicyNet`：`turn/response` 各自保留 state projection、action encoder 和 head，共享一段状态 encoder 主干，解决两个任务输入维度不同但仍需共享表征的问题。
- 训练入口已支持 `task: multitask`。v1.1 使用 epoch 级顺序调度；当前代码已新增 `batch_interleave`，会按有效训练 batch 进度比例交错更新 `turn` / `response`，valid 阶段仍分别评估两路指标。由于 response train 会过滤 pass-only 样本，P800 v1.2 显式配置 `task.interleave_batches`，避免用 response 原始行数高估调度权重。
- 多任务 checkpoint 可通过 `python -m mjai.cli eval --ckpt <ckpt> --task turn|response` 分别评估，也已兼容 `calibrate-response`、`sample-response-errors` 和 `sample-turn-errors`。
- P800 多任务训练已经完成多轮可对照结果：v1.1 epoch-sequential 证明 `turn` 受益但早期 `response` 弱于 improved；v1.2 batch-interleave 证明调度能改善 response macro/`daiminkan`，但没有解决核心 false/missed/F1 取舍；commit `7a7ad21` 的 v1.1 复训进一步刷新 20w turn overall。2026-06-24 的 40w v1.1 DDP run 刷新当前 turn 主候选，并让 response raw binary/multi 指标超过 06-04，但 exact `chi/pon/daiminkan` 和 missed call 仍不足以直接替换 05-10 improved 默认。默认多任务训练入口仍以 v1.1 为主，`batch_interleave` 作为显式实验能力保留，复现配置位于 `configs/experiments/`。DDP 已作为 P800 大数据训练路径跑通；自动 loss scale 与 value head 仍暂不混入当前变量。

下一阶段优先补：

- 40w 主候选的少数类错例聚合与服务端 load smoke
- DDP 吞吐、卡数和稳定性调优
- 训练过程中的动作分布、候选规模和非法标签质量闸门
- 类别不均衡下的自动权重估计、采样均衡和少数类错例聚合
- 更高吞吐的数据格式后续方向：当前已经完成 state/action 特征离线预编码，下一步应根据 `data_wait`、GPU 利用率和内存占用决定是否继续推进 mmap/Arrow streaming 数据集，而不是回到训练期规则特征重算。
- 根据错例聚合决定是否回补规则

数据重建与 P800 大数据训练提醒：

- turn/response action feature 维度已更新到当前数据版本，`runs/_datasets/tenhou_20w_2026-05-06/` 与 `runs/_datasets/tenhou_40w_2026-06-23/` 都已有独立 dataset evidence。后续只要 feature schema、rules 或 split 逻辑变化，就必须新建 dataset version 并重跑 `build-dataset` / `make-manifest`；只要 parser 或事件 schema 变化，就必须从 `preprocess` 开始重建。
- 20w baseline、improved 单任务、old v1.1 多任务、v1.2 `batch_interleave`、commit `7a7ad21` v1.1 复训、06-04 单任务 v1.2 safety 复训和 06-24 40w v1.1 DDP 训练都已跑完并归档。下一轮模型质量主线应在 40w `best_turn.pt` 少数类错例、40w response operating point 实战复核、`response_ce` loss scale 或少数类处理里一次只推进一个变量；DDP 后续作为吞吐/稳定性工程项单独推进，不要同时混入 value head 和复杂模型。
- 当前缺口不是“能否训练”：后续应继续按 `runs/<date>_<exp>/` 固化每轮模型证据，并通过 `dataset_id` 引用 `runs/_datasets/<dataset_id>/` 中的数据证据。
- 当前 response 默认 operating point 已切到 06-24 40w `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8`，用于极低错误鸣牌口径。05-10 improved `pass_bias=0.3` 与 06-24 40w `pass_bias=0.0 + no_yaku_call_penalty=0.4` 继续保留为稳定性和较高召回的对照口径。
- response 无役鸣牌已加入第一版推理侧安全约束：`calibrate-response` 已支持 raw / `pass_bias` / `no_yaku_call_penalty` 组合扫描，并报告 false call、missed call 和无明显役路 false call，避免只靠加大 `pass_bias` 牺牲正常副露召回。06-24 calibration 中 40w `best_response.pt + pass_bias=0.0 + no_yaku_call_penalty=0.4` 达到 test `false_call=0.884%`、`no_yaku_false=0.217%`、`missed=9.85%`、`F1=74.72%`、`multi top1=88.70%`；06-25 正式 valid 扩展扫描中当前默认档 `pass_bias=2.0 + no_yaku_call_penalty=0.8` 达到 `false_call=0.1972%`、`no_yaku_false=0.0083%`、`missed=31.52%`。默认档的代价是漏鸣明显升高，后续必须用人工/在线 case 复核这个取舍是否符合实际目标。

## 5. 阶段结论

如果按原始路线图划分，当前可以认为：

- `Task 1` 项目骨架初始化：已完成
- `Task 2` 牌表示与基础规则：已完成第一版
- `Task 3` 天凤解析与事件 schema：已完成
- `Task 4` 回放与合法动作生成：已完成第一版
- `Task 5` 样本构建：已完成第一版
- `preprocess / validate-events`：已完成第一版
- `Task 6` baseline 模型与训练引擎：已完成 `turn/response` 第一版，并已跑通 P800 大数据单任务 baseline
- `Task 7` 离线评估：已完成第一版
- `Task 8` 多任务训练：已完成可训练 baseline、P800 多轮验证和 40w DDP 训练；`turn` 正收益已确认并刷新主候选，`response` 仍需 operating point 实战复核和少数类质量改进
- `Task 9` 训练机适配：P800 单卡 baseline 与 DDP 40w 训练路径已验证；后续重点是多卡吞吐、卡数、稳定性和监控

## 6. 建议的下一步任务

建议按下面顺序推进：训练先行，规则边界按证据回补。

### 任务 1：实战复核当前 response 默认档

目标：

- 复核 06-24 40w `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8` 作为当前默认档后，错误鸣牌下降与漏鸣上升的取舍是否符合实战目标。

建议范围：

1. 当前默认：06-24 40w `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8`
2. 稳定历史 baseline：05-10 improved `pass_bias=0.3`
3. 较高召回对照：06-24 40w `best_response.pt + pass_bias=0.0 + no_yaku_call_penalty=0.4`
4. 后续多任务或新特征实验必须同时报告 raw、历史 baseline、较高召回对照和当前默认档指标

最低验收标准：

- valid/test 曲线一致。
- raw、历史 baseline、较高召回对照和当前默认档都有 false call、no-yaku false、missed call、non-pass F1、multi top1、`chi/pon` top1 记录。
- 推理默认值有明确配置位置或设计记录。

### 任务 2：response 无役鸣牌评估与定向错例抽样

建议范围：

1. 对当前默认档 `pass_bias=2.0 + no_yaku_call_penalty=0.8`、05-10 improved `pass_bias=0.3` 与 06-24 40w `pass_bias=0.0 + no_yaku_call_penalty=0.4` 各抽一份 `pass->chi`、`pass->pon`、`chi->pass`、`pon->pass`
2. `pass->chi/pass->pon` 重点判断误叫是否来自无役风险、防守/点况信息缺失、玩家风格差异或 action feature 不足
3. `chi->pass/pon->pass` 重点判断 40w 新候选是否误伤收益明确的副露，尤其是役牌、断幺、染手和对对和路线
4. 使用 `no_yaku_false_call_rate` / `no_yaku_false_call_share` 专门追踪 `features[38] == 1` 的 false call

### 任务 3：turn 防铳评估与风险 rerank

建议范围：

1. 新增离线防铳评估：预测弃牌是否在牌谱下一事件触发 `AGARI fromWho == actor`，并按立直家、庄家、失点分桶
2. 在服务端或离线代理中实现第一版 discard risk rerank：在对手立直/晚巡/庄家威胁下，对非现物、dora/赤、危险筋牌等候选施加风险惩罚
3. 后续再考虑训练 `discard_risk_head` 或 value/risk 辅助头，让 policy 学进攻合理性，risk head 学防铳代价

### 任务 4：继续规范 run 归档与质量闸门

建议范围：

1. 后续统一使用 `runs/<date>_<exp>/` 保存配置、日志、checkpoint 索引、eval report、calibration report 和错例样本
2. 数据集级证据统一放到 `runs/_datasets/<dataset_id>/`，run 通过 `dataset_id` 引用它
3. 固化非法标签复核结果，至少记录 `label_action not in legal_actions` 的 turn/response 计数
4. 在 run summary 中记录最终结论、默认 operating point、主要收益和已知回退

### 任务 5：规则边界按训练反馈回补

建议方向：

- 更完整的和牌机会放弃语义与残余振听边界
- 其他途中流局边界
- 更严格的立直限制
- 多家荣和等边界情况

触发条件：

- 某类规则缺口在大规模构建中导致高频失败或非法标签。
- 某类规则缺口在 eval 错例中形成稳定、可复现的主要误差来源。
