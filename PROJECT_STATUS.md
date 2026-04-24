# 项目进度状态

更新时间：`2026-04-24`

## 1. 总体判断

当前项目已经明显超过“纯规划阶段”，可以视为：

- `Phase 0` 基本完成：规则与数据的最小闭环已经跑通。
- `Phase 1` 尚未开始：特征、模型、训练、评估仍未落地。

换句话说，仓库当前最强的部分是“把牌谱可靠地变成合法训练样本”，而不是“训练模型”。

## 2. 已验证结果

以下结果已经在仓库内实际验证：

- `pytest` 全量通过，当前结果为 `37 passed`。
- `python -m mjai.cli build-dataset --input tests/fixtures/tenhou --output <tmp>` 可运行。
- 对 `tests/fixtures/tenhou` 中的 `10` 个 XML 夹具实测，生成：
  - `1542` 条 `turn` 样本
  - `4626` 条 `response` 样本
- `python -m mjai.cli make-manifest --input <tmp> --output <tmp>` 可运行。
- 同一批夹具可生成 manifest，实测分桶结果为：
  - `train_games=8`
  - `valid_games=1`
  - `test_games=0`

最后一项说明当前切分逻辑是按哈希稳定分桶，适合正式数据集；但在很小的样本集上，某个 split 为空是正常现象，后续最好补一个提示。

## 3. 按模块看当前状态

### 已完成或基本完成

- `mjai/data/tenhou/`
  - 天凤 XML 流式解析已实现。
  - 副露解码已实现，覆盖 `chi / pon / daiminkan / ankan / kakan`。
- `mjai/rules/`
  - 手牌状态、回放状态、弃牌/吃碰杠/荣和的第一版合法动作生成已实现。
  - 基础胡牌判断与立直前听牌检查已实现。
  - 回放校验器已实现，可验证弃牌合法性和副露消耗牌一致性。
- `mjai/data/builders/`
  - `turn` 与 `response` 样本构建器已实现。
  - Parquet 写出与 manifest 生成已实现。
- `configs/`
  - 数据、任务、模型、训练配置目录已建立。
- `tests/`
  - 规则、解析、样本构建、CLI、配置和数据写出都有测试覆盖。

### 部分完成

- `mjai/cli.py`
  - 子命令框架完整。
  - `build-dataset`、`make-manifest` 已可用。
  - `preprocess`、`validate-events`、`train`、`eval`、`replay-eval`、`inspect-sample`、`infer-on-state`、`dump-case` 仍是占位实现。
- 数据链路
  - 当前能从 XML 直接构建训练样本。
  - 但文档规划中的 `bronze_events` 持久化层还没有真正落地。

### 尚未开始或基本未开始

- `mjai/features/`
- `mjai/models/`
- `mjai/training/`
- `mjai/eval/`
- `mjai/serving/`

这些目录目前只有包骨架或说明性占位，还不能支持训练与推理闭环。

## 4. 当前最关键的差距

### 4.1 文档中的两阶段数据流水线尚未兑现

工程文档规划的是：

```text
raw_xml -> bronze_events -> silver_samples
```

但当前实际实现是：

```text
xml -> 解析 + 回放 + 样本构建 -> parquet samples
```

这意味着本地验证没问题，但想扩展到 20 万局甚至更大规模时，会缺少：

- 可复用的中间事件层
- 事件级校验报告
- 更稳定的数据版本管理
- 失败样本回溯能力

### 4.2 规则引擎仍是 first-pass 版本

当前规则层已经足够支撑“早期监督学习样本构建”，但还不足以称为完整日麻规则引擎。主要缺口包括：

- `tsumo`
- `ankan` / `kakan` 作为自家回合决策
- 完整振听判断
- `chankan`
- 多家荣和等边界情况
- 更完整的立直限制与流局处理

### 4.3 样本可训练，但训练链路还不存在

仓库已经能产出 `turn` 和 `response` 样本，但仍缺少：

- 状态特征编码
- 历史事件编码
- 候选动作编码
- baseline 模型
- loss / metrics
- 训练循环
- 离线评估报告

这也是当前项目最明显的阶段分界点。

## 5. 阶段结论

如果按原始路线图划分，当前可以认为：

- `Task 1` 项目骨架初始化：已完成
- `Task 2` 牌表示与基础规则：已完成
- `Task 3` 天凤解析与事件 schema：已完成
- `Task 4` 回放与合法动作生成：已完成第一版
- `Task 5` 样本构建：已完成第一版
- `Task 6` baseline 模型与训练引擎：未开始
- `Task 7` 离线评估：未开始
- `Task 8` 多任务训练：未开始
- `Task 9` 训练机适配：只完成配置骨架，未验证训练链路

## 6. 建议的下一步任务

建议按下面顺序推进，不要同时铺太多面。

### 任务 1：补齐 `preprocess` 与 `validate-events`

目标：

- 把当前“直接从 XML 构建样本”的实现拆成真正的两阶段流水线。
- 产出可持久化的 `bronze_events`。
- 为失败牌谱和异常事件生成报告。

最低验收标准：

- `python -m mjai.cli preprocess` 可把 XML 转成事件文件。
- `python -m mjai.cli validate-events` 可输出成功率、失败数和错误摘要。
- `build-dataset` 可以从 `bronze_events` 而不是原始 XML 继续构建样本。

### 任务 2：补充数据统计与样本诊断

目标：

- 在构建样本后直接看到动作分布、标签分布、空 split、非法标签率等信息。
- 为后续 `response` 任务的类别不平衡处理提前准备依据。

最低验收标准：

- 输出每种动作的样本数统计。
- 输出每个 split 的局数和样本数。
- 小数据集出现空 split 时给出提示而不是静默通过。

### 任务 3：实现最小可训练闭环

目标：

- 先打通 `turn_head` baseline，不要一开始就上完整多任务。

建议范围：

- `state/history/action` 三类特征编码的最小版本
- 一个轻量共享编码器或直接的 baseline ranker
- `turn` 任务 loss 与 accuracy
- 本地 `4070` smoke test

最低验收标准：

- 能读取 manifest 和 Parquet 样本。
- 能跑完 `1` 个 epoch。
- loss 可以下降。
- 训练过程中不出现非法候选或 NaN。

### 任务 4：扩展规则边界，优先补高价值动作

建议优先级：

1. `tsumo`
2. `ankan / kakan` 的自家回合决策
3. 更完整的振听与 `ron` 过滤
4. 更严格的立直限制

原因很直接：这些动作会直接影响标签空间定义，最好在大规模训练前补稳。
