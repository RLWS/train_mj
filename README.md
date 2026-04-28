# 雀魂四麻 AI 训练仓库

本仓库用于构建基于天凤牌谱的雀魂四麻 AI 训练链路，目标是形成一条可验证、可训练、可扩展的工程闭环：

```text
天凤 XML -> 事件解析 -> 状态回放 -> 合法动作生成 -> 样本构建 -> 模型训练 -> 推理校验
```

## 当前状态

- 已完成第一阶段的数据闭环：天凤 XML 解析、`bronze_events` 持久化、回放校验、合法动作生成、`turn/response` 样本构建、Parquet 写出、按 `game_id` 的 manifest 切分。
- 已补上第一版诊断输出：事件分布、失败原因聚合、动作标签分布、split 样本数和空 split warning。
- 已打通第一版最小训练闭环：`turn/response` manifest 读取、状态/动作特征、候选排序模型、单 epoch smoke train、checkpoint 与指标落盘。
- 已补上第一版离线评估：可从 checkpoint 对 `turn/response` 的 `train/valid/test` split 做 `loss`、`top-k accuracy` 和按动作类别聚合评估，并导出 JSON 报告。
- 已补上第一版样本抽查：可直接从 `silver_samples` 目录按全局索引读取样本，查看 label、合法动作和状态快照。
- 已补上第一版 case 定位：可从 `eval` 错例跳转到 `inspect-sample`，再按 `game_id / kyoku_index / step_index` 回溯 `bronze_events` 上下文。
- 已补上第一版自家高价值回合动作：`turn` 合法动作与样本链路已支持 `tsumo`、`ankan` 和 `kakan`，其中当前真实夹具已覆盖 `tsumo` 与 `ankan` 样本。
- 已补上第一版 `ron` 正确性过滤：当前 `response` 合法动作与样本链路已支持自家舍牌振听、同巡振听和立直后振听。
- 当前测试基线为 `pytest` 全量通过，结果是 `72 passed`。
- 使用 `tests/fixtures/tenhou` 实测可完成：
  - `10` 局 XML 预处理
  - `3253` 条事件写入 `bronze_events`
  - `10` 个事件分片全部校验通过
  - `1556` 条 `turn` 样本
  - `4626` 条 `response` 样本
  - `turn` 单任务 `1` epoch smoke train
  - `response` 单任务 `1` epoch smoke train
  - `turn` / `response` checkpoint 的 `valid` split 离线评估
- `serving/` 仍基本是骨架或占位；`features/`、`models/`、`training/`、`eval/` 已有最小 `turn/response` baseline 实现。

## 当前可用命令

```bash
python -m mjai.cli preprocess --input <raw_xml_dir> --output <bronze_events_dir>
python -m mjai.cli validate-events --input <bronze_events_dir> --report <report_json>
python -m mjai.cli build-dataset --input <bronze_events_dir> --output <samples_dir>
python -m mjai.cli make-manifest --input <samples_dir> --output <manifest_dir>
python -m mjai.cli train --config <train_config_yaml>
python -m mjai.cli eval --ckpt <checkpoint_path> --split valid --report <eval_report_json>
python -m mjai.cli inspect-sample --dataset <samples_dir> --index 100
python -m mjai.cli inspect-sample --dataset <samples_dir> --sample-id <sample_id>
python -m mjai.cli dump-case --game-id <game_id> --kyoku-id <kyoku_index> --step-id <step_index>
pytest
```

说明：

- `preprocess` 会把天凤 XML 转成事件分片 Parquet，并输出 `preprocess_summary.json`。
- `validate-events` 会回放校验事件分片，并输出包含事件分布、失败原因和回放摘要的 JSON 报告。
- `build-dataset` 当前从 `bronze_events` 构建 `turn/response` Parquet 样本，并输出标签分布摘要；当前真实夹具的 `turn` 标签已覆盖 `discard / riichi_discard / tsumo / ankan`，构建链路本身已支持 `kakan`。
- `make-manifest` 会按 `game_id` 生成 `train/valid/test` 清单，并输出 split 样本统计与 warning。
- `train` 当前支持最小 `turn/response` baseline，并输出 `config.resolved.yaml`、`train_metrics.json`、`best.pt`、`final.pt`。
- `eval` 当前支持最小离线评估，输出 `loss`、`top1/top3/top5 accuracy`、按动作类别聚合指标、候选规模分桶和错例导出，并生成 JSON 报告。
- `inspect-sample` 当前支持从样本目录或单个 parquet 文件按索引或 `sample_id` 抽查样本。
- `dump-case` 当前支持从 `bronze_events` 中定位目标决策点的原始事件上下文。
- `replay-eval`、`infer-on-state` 仍未实装。

## 当前缺口

- 规则引擎还不是完整日麻规则实现；虽然已补上自家舍牌振听、同巡振听和立直后振听的 `ron` 过滤，但仍缺少更完整的和牌机会放弃语义、`chankan`、`kyuushu_kyuuhai`、途中流局与完整计分结算等能力。
- 诊断输出已经有第一版，并已打通 `eval -> inspect-sample -> dump-case` 的轻量跳转链路，但还缺少更细的错例聚合、固定 case 集合和更丰富的数据质量报告。
- 当前训练与评估只覆盖最小 `turn/response` baseline；多任务、错误分析工具和推理服务都还没开始。

## 文档索引

- `MAHJONG_AI_PLAN.md`：总体方案与训练路线。
- `MAHJONG_AI_ENGINEERING.md`：工程设计与目录、CLI、配置约定。
- `PROJECT_STATUS.md`：当前实现进度、已验证结果、主要缺口和建议的下一步任务。
- `RULE_COVERAGE.md`：当前规则引擎覆盖范围与未实现边界。

## 建议的下一步

1. 先继续补错误分析深度，围绕现有 `eval -> inspect-sample -> dump-case` 链路补固定错例集和更细的错例聚合。
2. 然后补规则边界，优先做更完整的振听/`ron` 限制、`chankan` 和流局边界。
3. 最后继续扩训练覆盖，补多任务训练、更完整离线指标和推理入口。
