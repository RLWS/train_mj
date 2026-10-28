# 完整状态 v2：训练、评估与接入

更新时间：2026-10-28。

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

2026-10-28 已完成训练产物、独立 valid/test 和扩展服务参数校准分析，离线指标总体改善；尚无实战牌力改善证据。详见 [本轮最终分析](training-analysis-2026-10-28.md)。
现有合法动作生成、计分及多家响应边界仍然存在，本轮没有实现完整模拟器。

## 隔离与兼容性

本轮改造已收尾合回主线：`configs/api_local.yaml` 现在指向归档的 full_state_v2 checkpoint（8000 端口），
旧 2026-06-24 模型仍保留在 `runs/2026-06-24_p800_multitask_v1_1_ddp_40w/`，可随时切回。
服务不自动启动，也不连接 P800。

建议使用独立 Python 环境；不要在实战共享环境中执行指向实验 checkout 的 `pip install -e .`。
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

默认 eval 不启用 DDP，以下命令单进程执行。

**2026-10-28 修正：首轮 checkpoint 内的 test_manifest 继承了旧的 `data/manifests/test.jsonl`。**
当前配置已补上 `data/full_state_v2/manifests/test.jsonl`。已训练 checkpoint 不会随配置文件自动改变，
所以下面的 test 命令显式传入 `--config` 覆盖路径。三条 valid eval/校准命令不受此问题影响。

```bash
python -m mjai.cli eval --ckpt runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt --task turn --split valid --report outputs/reports/full_state_v2_turn_valid.json
python -m mjai.cli eval --ckpt runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_response.pt --task response --split valid --report outputs/reports/full_state_v2_response_valid.json
python -m mjai.cli calibrate-response --ckpt runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_response.pt --split valid --report outputs/reports/full_state_v2_response_calibration_valid.json
python -m mjai.cli eval --ckpt runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_turn.pt --task turn --split test --config configs/multitask_full_state_p800.yaml --report outputs/reports/full_state_v2_turn_test.json
python -m mjai.cli eval --ckpt runs/2026-10-28_p800_multitask_full_state_v2/multitask/checkpoints/best_response.pt --task response --split test --config configs/multitask_full_state_p800.yaml --report outputs/reports/full_state_v2_response_test.json
```

在 valid 上选择参数，然后固定参数在 test 上报告；不要反复根据 test 挑参数。
本轮扩展校准在 valid 网格选出 `pass_bias=1.5`、`no_yaku_call_penalty=0`，`configs/api_local.yaml` 已采用；
配置中的 `recall=1.0`、`conservative=2.0` 对照档即来自校准报告的召回/保守候选。

对比 baseline 时固定原始 game_id 集合和 split。原模型在原 checkout/旧 schema 上评估，
新模型在本分支/新 schema 上评估，再按相同 game/sample 身份对齐报告。
记录 turn top1/top3、立直弃牌、杠类、response multi-candidate、false/missed、F1、
合法标签验证结果以及实际对局收益；候选有役与仲裁不完整的边界仍需单独审查。

## 新 API 接入

训练产物已归档，可直接启动：

```bash
python -m mjai.cli serve-api --config configs/api_local.yaml
```

地址为 `127.0.0.1:8000`，默认加载归档的 full_state_v2 checkpoint，默认 operating point 为
`pass_bias=1.5`、`no_yaku_call_penalty=0`。请求字段契约、校验规则与已知训练侧边界见
[full-state-api.md](full-state-api.md)，完整向量布局见 [model-inputs.md](model-inputs.md)。
实战客户端切换前先完成新字段适配与独立对比，不要把生产流量直接切到新模型。

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
