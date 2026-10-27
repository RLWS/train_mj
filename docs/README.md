# 文档索引

本目录存放项目的长期文档。根目录只保留 `README.md` 作为入口，以及 `AGENTS.md` 作为协作约定。

新增实验分支入口：[完整状态 v2 训练流程与命令](full-state-training.md)。

## 推荐阅读顺序

1. [status.md](status.md)：当前项目状态、已验证结果、主要缺口和下一步任务。
2. [engineering-guide.md](engineering-guide.md)：当前真实工程布局、配置、CLI、数据 schema、训练与 serving 边界。
3. [rule-coverage.md](rule-coverage.md)：规则引擎当前已覆盖和明确未覆盖的边界。
4. [releases/v1.2.0.md](releases/v1.2.0.md)：v1.2.0 标签对应的 DDP、40w、分布式评估和保守 response 默认档。
5. [smoke-training.md](smoke-training.md)：用夹具数据跑通预处理、构建样本、训练、评估和调试链路。
6. [multitask-training.md](multitask-training.md)：多任务模型结构、P800/DDP 启动方式、checkpoint、评估口径和 40w run 结论。
7. [model-inputs.md](model-inputs.md)：当前 turn / response 模型输入张量、维度和特征语义。
8. [serving-api-plan.md](serving-api-plan.md)：已实现 serving/API 的分层设计、接口约定和剩余阶段。
9. [training-analysis-2026-06-04.md](training-analysis-2026-06-04.md)：2026-06-04 P800 单任务复训历史快照。
10. [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md)：2026-05 baseline/improved 与特征实验历史快照。
11. [architecture-plan.md](architecture-plan.md)：总体 AI、数据、模型和长期训练路线。
12. [releases/v1.1.0.md](releases/v1.1.0.md)：2026-05-17 的 v1.1.0 发布快照。

## 命名说明

- `architecture-plan.md` 原为 `MAHJONG_AI_PLAN.md`，保留长期方案和训练路线；当前实现以 engineering/status 为准。
- `engineering-guide.md` 原为 `MAHJONG_AI_ENGINEERING.md`，现作为真实工程结构和 CLI 约定。
- `status.md` 原为 `PROJECT_STATUS.md`，作为当前事实来源。
- `rule-coverage.md` 原为 `RULE_COVERAGE.md`，记录规则边界。
- `smoke-training.md` 原为 `SMOKE_TRAIN_TUTORIAL.md`，记录本地 smoke 工作流。
- `model-inputs.md` 记录当前模型输入 schema；修改 state/action feature 维度时必须同步更新。
- `multitask-training.md` 记录 v1.1 多任务训练主线；多任务和 DDP 实验结论优先写这里，不再继续扩张单任务训练分析文档。
- `serving-api-plan.md` 记录 serving/API 模块方案与实现状态；维护或扩展 API 时应同步其分层边界和 response operating point。
- `training-analysis-2026-06-04.md` 记录 2026-06-04 P800 单任务复训与无役鸣牌 calibration，作为当前最新单任务训练结论。
- `releases/` 记录标签对应的 release notes；历史数字只代表对应发布日期，不作为当前状态。
