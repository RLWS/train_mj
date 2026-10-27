# 完整状态 v2：训练、评估与接入

更新时间：2026-10-27。实验分支：`codex/full-state-features`。

## 实验范围

本轮保持 MLP 候选排序结构、交叉熵目标和正式配置的主要超参数，改进输入信息：

- 四家完整副露（含赤牌、来源、暗杠和加杠后的组成）。
- 场风、场内局号、东风/半庄、计划末局/延长场、相对点差。
- 摸切/手切、立直宣言牌、公开鸣牌/立直/宝牌事件与弃牌顺序的关联。
- 按实际摸牌事件统计四麻剩余摸牌机会，取代按自己弃牌数估算。
- state 和 candidate 同步使用自家相对座位，返回动作仍为平台座位。
- snapshot/API 统一使用实际点数（25000），修正原训练快照的百点单位偏差。
- 保留重复宝牌指示牌的计数；立直确认后更新供托。

state 维度 turn/response 从 8173/8216 变为 **11969/12012**；candidate 仍为 89/91。
输入 float32 的未压缩体积约增加 46%，实际 Parquet 压缩比与吞吐需实测。
正式配置保持 hidden_size=512、epoch_sequential 和原类别权重；DDP batch size 先沿用旧值，
如显存不足再单独减小并记录差异。新增特征与单位修复是一个组合实验，不能将收益归因于其中单项。

这是可训练的工程版本，尚无大数据复训或实战牌力改善证据。
现有合法动作生成、计分及多家响应边界仍然存在，本轮没有实现完整模拟器。

## 隔离与兼容性

所有命令在**新 worktree 根目录**运行。原实战 checkout、分支、API 配置和已有模型继续保持原样。
本分支不自动启动服务、不连接 P800、不执行远程训练。实验 API 使用 8001 端口。

建议使用独立 Python 环境；不要在实战共享环境中执行指向新 worktree 的 `pip install -e .`。
本项目依赖可在独立环境安装 `python -m pip install -e ".[dev,api]"`；PyTorch 需按设备另行安装，
P800 继续使用已验证的厂商 PyTorch 环境。

**不能复用旧 bronze、silver、manifest 或 checkpoint。**
必须从 XML 重新 preprocess，因为旧摸切标记和赛制信息不满足新契约。
build-dataset 拒绝不带 `full_state_v2` 标记的 INIT；eval、校准、错例抽样及 serving
拒绝旧 checkpoint。预编码 state 字节长度检查也会阻止旧 silver 被本分支读取。
旧模型仍可通过原 checkout 使用，不做自动权重迁移。

## 本地 RTX 4070 smoke

下面命令兼容 PowerShell 和 Bash，每条成功后再执行下一条。输出使用独立的 smoke 目录。

```bash
python -m mjai.cli preprocess --input tests/fixtures/tenhou --output data/full_state_v2_smoke/bronze_events
python -m mjai.cli validate-events --input data/full_state_v2_smoke/bronze_events --report outputs/reports/full_state_v2_smoke_events.json
python -m mjai.cli build-dataset --input data/full_state_v2_smoke/bronze_events --output data/full_state_v2_smoke/silver_samples --workers 2 --skip-validation
python -m mjai.cli make-manifest --input data/full_state_v2_smoke/silver_samples --output data/full_state_v2_smoke/manifests
python -m mjai.cli train --config configs/turn_full_state_smoke.yaml
python -m mjai.cli train --config configs/response_full_state_smoke.yaml
python -m mjai.cli eval --ckpt outputs/checkpoints/turn_full_state_smoke/best.pt --split valid --report outputs/reports/turn_full_state_smoke_valid.json
python -m mjai.cli eval --ckpt outputs/checkpoints/response_full_state_smoke/best_response.pt --split valid --report outputs/reports/response_full_state_smoke_valid.json
python -m mjai.cli calibrate-response --ckpt outputs/checkpoints/response_full_state_smoke/best_response.pt --split valid --report outputs/reports/response_full_state_smoke_calibration.json
```

只有 validate-events 报告 `files_failed=0` 后才使用 `--skip-validation`。
两个 smoke 配置均为 CUDA、hidden_size=64、1 epoch，仅用于链路验证。

## P800 正式训练

先将本分支代码放入独立 checkout/容器代码目录，使用原始 XML 的只读输入路径。
`<RAW_XML_DIR>` 需要替换为真实目录；其余输出路径属于新 checkout。
不要使用旧 `start.sh` 的默认目录串联此次实验。

```bash
python -m mjai.cli preprocess --input <RAW_XML_DIR> --output data/full_state_v2/bronze_events
python -m mjai.cli validate-events --input data/full_state_v2/bronze_events --report outputs/reports/full_state_v2_events.json
python -m mjai.cli build-dataset --input data/full_state_v2/bronze_events --output data/full_state_v2/silver_samples --workers 32 --skip-validation
python -m mjai.cli make-manifest --input data/full_state_v2/silver_samples --output data/full_state_v2/manifests
python -m torch.distributed.run --standalone --nproc_per_node=8 -m mjai.cli train --config configs/multitask_full_state_p800.yaml
```

`--nproc_per_node` 按实际可用卡数调整。正式配置沿用项目已有 `backend: vendor`，不适用于普通 CUDA DDP 环境。
数据量增加前先测构建吞吐与磁盘空间；workers 从 32 起，按实测再决定是否增加。
如需重新构建另一版数据，用新的目录，避免将不同 schema/实验的分片混在一起。

## 正式评估与校准

checkpoint 内保存正式 manifest 路径；默认 eval 不启用 DDP，以下命令单进程执行。

```bash
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_full_state_v2_ddp/best_turn.pt --task turn --split valid --report outputs/reports/full_state_v2_turn_valid.json
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_full_state_v2_ddp/best_response.pt --task response --split valid --report outputs/reports/full_state_v2_response_valid.json
python -m mjai.cli calibrate-response --ckpt outputs/checkpoints/multitask_full_state_v2_ddp/best_response.pt --split valid --report outputs/reports/full_state_v2_response_calibration_valid.json
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_full_state_v2_ddp/best_turn.pt --task turn --split test --report outputs/reports/full_state_v2_turn_test.json
python -m mjai.cli eval --ckpt outputs/checkpoints/multitask_full_state_v2_ddp/best_response.pt --task response --split test --report outputs/reports/full_state_v2_response_test.json
```

在 valid 上选择参数，然后固定参数在 test 上报告；不要反复根据 test 挑参数。
更改模型后旧 pass_bias 的效果不保证保持一致。`api_full_state.yaml` 暂保留原保守参数以控制变量，
它不是为新模型重新验证过的 operating point。

对比 baseline 时固定原始 game_id 集合和 split。原模型在原 checkout/旧 schema 上评估，
新模型在本分支/新 schema 上评估，再按相同 game/sample 身份对齐报告。
记录 turn top1/top3、立直弃牌、杠类、response multi-candidate、false/missed、F1、
合法标签验证结果以及实际对局收益；候选有役与仲裁不完整的边界仍需单独审查。

## 新 API 接入契约

大数据训练完成并准备独立验证时，可手动启动：

```bash
python -m mjai.cli serve-api --config configs/api_full_state.yaml
```

地址为 `127.0.0.1:8001`。不应将实战客户端直接切到这里：调用方先适配新字段。

- `feature_schema: "full_state_v2"`。
- `scores` 为实际点数；actor、dealer、source 及四家数组仍为平台绝对座位。
- `round_wind`、`round_number` 均为 0..3，后者为一至四局，不能把连庄次数算进局号。
- `match_rounds` 为 4（东风）、8（半庄）或 null（确实未知）。
- `remaining_tiles` 为四麻剩余摸牌机会，初始 70；包含岭上摸牌的消耗。
- `all_melds` 必须有四家，`own_melds` 必须与 `all_melds[actor]` 一致。
- `discard_history` 每项包含 actor、tile、tsumogiri（布尔或 null）、riichi_discard（布尔）。
- `public_events` 保存 reach1/reach2/chi/pon/daiminkan/ankan/kakan/dora；每项带 discard_count。
  reach 需 actor；鸣牌需 actor/source/tile；dora 需 tile。discard_count 必须按时间非递减。
- 必须提供四家立直标记/巡目、牌河、宝牌指示牌、手牌、本场、供托、自家实际摸牌次数等基础字段。
- turn 选择立直前的请求不能预先写入本次 reach1；对手已经发生的 reach1 则应保留。
- 不提供对手隐藏手牌、摸牌牌面或未来事件。缺失必需字段返回 422，不用零值静默补齐。

完整向量布局见 [model-inputs.md](model-inputs.md)。

## 本地验证记录

2026-10-27，RTX 4070 Laptop GPU / PyTorch 2.5.1+cu121：

- 21 个 XML → 7386 个事件；21/21 校验通过。
- 3532 turn / 10506 response 样本；按 game_id 切分为 19/1/1 个 game。
- turn CUDA smoke：1 epoch，valid 155 样本，loss 1.4998、top1 52.26%、top3 83.87%。
- response CUDA smoke：1 epoch，valid 465 样本，loss 0.1534、overall top1 96.34%；
  multi-candidate 只有 31 样本，top1 45.16%。overall 被 pass-only 样本明显抬高。
- 两路 checkpoint 均已完成离线 eval，response 已完成校准；两路已训练 checkpoint 也通过真实局面的推理检查。
- 新增测试覆盖座位旋转、可见信息差异、物理牌 ID 摸切识别、隐藏摸牌不泄漏、
  立直动作不泄漏、真实牌谱训练/serving 编码一致、旧 bronze/checkpoint 拒绝及新 API 缺字段拒绝。

全量回归：`python -m pytest -q`，145 passed；最后 API 字段一致性调整后的定向回归 22 passed。
`compileall`、`git diff --check` 和本页/特征文档链接检查通过。
这些小样本数字只证明流程可运行，不能作为牌力提升结论。

赛制解析参考：[tenhou-log-utils GO 位掩码实现](https://github.com/mthrok/tenhou-log-utils/blob/master/tenhou_log_utils/parser.py)；半庄使用 `0x08`。
