# 雀魂四麻 AI 训练仓库

本仓库用于构建基于天凤牌谱的雀魂四麻 AI 训练链路，目标是形成一条可验证、可训练、可扩展的工程闭环：

```text
天凤 XML -> 事件解析 -> 状态回放 -> 合法动作生成 -> 样本构建 -> 模型训练 -> 推理校验
```

## 当前状态

- 已完成第一阶段的数据闭环：天凤 XML 解析、`bronze_events` 持久化、回放校验、合法动作生成、`turn/response` 样本构建、Parquet 写出、按 `game_id` 的 manifest 切分。
- 当前测试基线为 `pytest` 全量通过，结果是 `40 passed`。
- 使用 `tests/fixtures/tenhou` 实测可完成：
  - `10` 局 XML 预处理
  - `3253` 条事件写入 `bronze_events`
  - `10` 个事件分片全部校验通过
  - `1542` 条 `turn` 样本
  - `4626` 条 `response` 样本
- `features/`、`models/`、`training/`、`eval/`、`serving/` 仍基本是骨架或占位，训练链路尚未启动。

## 当前可用命令

```bash
python -m mjai.cli preprocess --input <raw_xml_dir> --output <bronze_events_dir>
python -m mjai.cli validate-events --input <bronze_events_dir> --report <report_json>
python -m mjai.cli build-dataset --input <bronze_events_dir> --output <samples_dir>
python -m mjai.cli make-manifest --input <samples_dir> --output <manifest_dir>
pytest
```

说明：

- `preprocess` 会把天凤 XML 转成事件分片 Parquet，并输出 `preprocess_summary.json`。
- `validate-events` 会回放校验事件分片，并输出 JSON 报告。
- `build-dataset` 当前从 `bronze_events` 构建 `turn/response` Parquet 样本。
- `make-manifest` 会按 `game_id` 生成 `train/valid/test` 清单。
- `train`、`eval`、`replay-eval`、`inspect-sample`、`infer-on-state`、`dump-case` 仍未实装。

## 当前缺口

- 规则引擎还不是完整日麻规则实现，缺少 `tsumo`、`ankan/kakan` 的回合决策、完整振听判断等能力。
- 事件校验报告和样本统计目前还是最小版本，缺少动作分布、失败类型归类、空 split 提示等诊断信息。
- 训练、评估、推理服务和特征编码模块还未进入实际开发。

## 文档索引

- `MAHJONG_AI_PLAN.md`：总体方案与训练路线。
- `MAHJONG_AI_ENGINEERING.md`：工程设计与目录、CLI、配置约定。
- `PROJECT_STATUS.md`：当前实现进度、已验证结果、主要缺口和建议的下一步任务。
- `RULE_COVERAGE.md`：当前规则引擎覆盖范围与未实现边界。

## 建议的下一步

1. 先补数据统计与诊断输出，补齐动作分布、split 分布、失败摘要和空 split 提示。
2. 再实现最小可训练闭环：特征编码、`turn_head` baseline、训练循环和本地 smoke test。
3. 然后扩展响应动作与规则边界，处理 `pass` 不平衡和更多日麻细节。
