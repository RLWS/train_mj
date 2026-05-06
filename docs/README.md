# 文档索引

本目录存放项目的长期文档。根目录只保留 `README.md` 作为入口，以及 `AGENTS.md` 作为协作约定。

## 推荐阅读顺序

1. [status.md](status.md)：当前项目状态、已验证结果、主要缺口和下一步任务。
2. [smoke-training.md](smoke-training.md)：用夹具数据跑通预处理、构建样本、训练、评估和调试链路。
3. [model-inputs.md](model-inputs.md)：当前 turn / response 模型输入张量、维度和特征语义。
4. [training-analysis-2026-05-02.md](training-analysis-2026-05-02.md)：turn / response 训练效果分析、response 特征增强记录，以及 2026-05-06 P800 大数据 baseline 结论。
5. [rule-coverage.md](rule-coverage.md)：规则引擎当前已覆盖和明确未覆盖的边界。
6. [architecture-plan.md](architecture-plan.md)：总体 AI、数据、模型和训练路线。
7. [engineering-guide.md](engineering-guide.md)：工程布局、配置、CLI、数据格式和任务计划。

## 命名说明

- `architecture-plan.md` 原为 `MAHJONG_AI_PLAN.md`，保留总体方案和训练路线细节。
- `engineering-guide.md` 原为 `MAHJONG_AI_ENGINEERING.md`，保留工程落地和 CLI 约定。
- `status.md` 原为 `PROJECT_STATUS.md`，作为当前事实来源。
- `rule-coverage.md` 原为 `RULE_COVERAGE.md`，记录规则边界。
- `smoke-training.md` 原为 `SMOKE_TRAIN_TUTORIAL.md`，记录本地 smoke 工作流。
- `model-inputs.md` 记录当前模型输入 schema；修改 state/action feature 维度时必须同步更新。
