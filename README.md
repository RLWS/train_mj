# 雀魂四麻 AI 训练仓库

本仓库用于构建基于天凤牌谱的雀魂四麻 AI 训练链路，目标是形成一条可验证、可训练、可扩展的工程闭环：

```text
天凤 XML -> 事件解析 -> 状态回放 -> 合法动作生成 -> 样本构建 -> 模型训练 -> 推理校验
```

## 当前状态

- 已完成第一阶段的数据闭环：天凤 XML 解析、`bronze_events` 持久化、回放校验、合法动作生成、`turn/response` 样本构建、Parquet 写出、按 `game_id` 的 manifest 切分。
- 已补上第一版诊断输出：事件分布、失败原因聚合、动作标签分布、split 样本数和空 split warning。
- 已打通第一版最小训练闭环：`turn/response` manifest 读取、状态/动作特征、候选排序模型、单 epoch smoke train、checkpoint 与指标落盘。
- 当前测试基线为 `pytest` 全量通过，结果是 `47 passed`。
- 使用 `tests/fixtures/tenhou` 实测可完成：
  - `10` 局 XML 预处理
  - `3253` 条事件写入 `bronze_events`
  - `10` 个事件分片全部校验通过
  - `1542` 条 `turn` 样本
  - `4626` 条 `response` 样本
  - `turn` 单任务 `1` epoch smoke train
  - `response` 单任务 `1` epoch smoke train
- `eval/`、`serving/` 仍基本是骨架或占位；`features/`、`models/`、`training/` 已有最小 `turn/response` baseline 实现。

## 当前可用命令

```bash
python -m mjai.cli preprocess --input <raw_xml_dir> --output <bronze_events_dir>
python -m mjai.cli validate-events --input <bronze_events_dir> --report <report_json>
python -m mjai.cli build-dataset --input <bronze_events_dir> --output <samples_dir>
python -m mjai.cli make-manifest --input <samples_dir> --output <manifest_dir>
python -m mjai.cli train --config <train_config_yaml>
pytest
```

说明：

- `preprocess` 会把天凤 XML 转成事件分片 Parquet，并输出 `preprocess_summary.json`。
- `validate-events` 会回放校验事件分片，并输出包含事件分布、失败原因和回放摘要的 JSON 报告。
- `build-dataset` 当前从 `bronze_events` 构建 `turn/response` Parquet 样本，并输出标签分布摘要。
- `make-manifest` 会按 `game_id` 生成 `train/valid/test` 清单，并输出 split 样本统计与 warning。
- `train` 当前支持最小 `turn/response` baseline，并输出 `config.resolved.yaml`、`train_metrics.json`、`best.pt`、`final.pt`。
- `eval`、`replay-eval`、`inspect-sample`、`infer-on-state`、`dump-case` 仍未实装。

## 当前缺口

- 规则引擎还不是完整日麻规则实现，缺少 `tsumo`、`ankan/kakan` 的回合决策、完整振听判断等能力。
- 诊断输出已经有第一版，但还缺少更细的 case drill-down、失败样本导出和更丰富的数据质量报告。
- 当前训练只覆盖最小 `turn/response` baseline，多任务、离线评估和推理服务都还没开始。

## 文档索引

- `MAHJONG_AI_PLAN.md`：总体方案与训练路线。
- `MAHJONG_AI_ENGINEERING.md`：工程设计与目录、CLI、配置约定。
- `PROJECT_STATUS.md`：当前实现进度、已验证结果、主要缺口和建议的下一步任务。
- `RULE_COVERAGE.md`：当前规则引擎覆盖范围与未实现边界。

## 建议的下一步

1. 先补规则边界，优先做 `tsumo`、`ankan/kakan` 回合动作和更完整的 `ron/riichi` 限制。
2. 然后补离线评估和错误分析，至少有 `turn/response` 的 top-k、分类别指标和报告导出。
3. 最后继续增强诊断与抽查，补失败 case 导出、样本抽查和更细粒度报告。
