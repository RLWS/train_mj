# 项目进度状态

更新时间：`2026-04-24`

## 1. 总体判断

当前项目已经明显超过“纯规划阶段”，可以视为：

- `Phase 0` 第一版已完成：规则与数据的最小闭环已经跑通。
- `Phase 1` 尚未开始：特征、模型、训练、评估仍未落地。

换句话说，仓库当前最强的部分是“把牌谱可靠地变成合法训练样本”，而不是“训练模型”。

## 2. 已验证结果

以下结果已经在仓库内实际验证：

- `pytest` 全量通过，当前结果为 `41 passed`。
- `python -m mjai.cli preprocess --input tests/fixtures/tenhou --output <tmp>` 可运行。
- 对 `tests/fixtures/tenhou` 中的 `10` 个 XML 夹具实测，生成 `3253` 条事件并写入 `bronze_events`。
- `python -m mjai.cli validate-events --input <tmp> --report <tmp>` 可运行。
- 同一批事件分片实测回放结果为：
  - `files=10`
  - `valid=10`
  - `failed=0`
  - `total_events=3253`
  - 可输出事件分布、失败原因聚合、总局数/弃牌数/副露数摘要
- `python -m mjai.cli build-dataset --input <tmp> --output <tmp>` 可运行。
- 对 `tests/fixtures/tenhou` 中的 `10` 个 XML 夹具实测，生成：
  - `1542` 条 `turn` 样本
  - `4626` 条 `response` 样本
  - 可输出 `turn/response` 标签动作分布
- `python -m mjai.cli make-manifest --input <tmp> --output <tmp>` 可运行。
- 同一批夹具可生成 manifest，实测分桶结果为：
  - `train_games=9`
  - `valid_games=1`
  - `test_games=0`
  - 当前会对空 split 输出 warning

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
  - `bronze_events` 预处理器已实现。
  - 事件分片回放校验器已实现。
  - `turn` 与 `response` 样本构建器已实现。
  - Parquet 写出与 manifest 生成已实现。
  - 事件分布、动作分布、split 样本统计和 warning 已实现第一版。
- `configs/`
  - 数据、任务、模型、训练配置目录已建立。
- `tests/`
  - 规则、解析、样本构建、CLI、配置和数据写出都有测试覆盖。

### 部分完成

- `mjai/cli.py`
  - 子命令框架完整。
  - `preprocess`、`validate-events`、`build-dataset`、`make-manifest` 已可用。
  - `train`、`eval`、`replay-eval`、`inspect-sample`、`infer-on-state`、`dump-case` 仍是占位实现。
- 数据链路
  - 已经落地 `raw_xml -> bronze_events -> silver_samples` 两阶段流水线。
  - 第一版诊断输出已经落地，但还缺少更细的失败样本导出和 case drill-down。

### 尚未开始或基本未开始

- `mjai/features/`
- `mjai/models/`
- `mjai/training/`
- `mjai/eval/`
- `mjai/serving/`

这些目录目前只有包骨架或说明性占位，还不能支持训练与推理闭环。

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

但要把它作为大规模训练前的数据入口，仍然建议继续补：

- 更方便的失败 case 回溯
- 样本级抽查工具
- 更细的分片级和动作级质量报告

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
- `preprocess / validate-events`：已完成第一版
- `Task 6` baseline 模型与训练引擎：未开始
- `Task 7` 离线评估：未开始
- `Task 8` 多任务训练：未开始
- `Task 9` 训练机适配：只完成配置骨架，未验证训练链路

## 6. 建议的下一步任务

建议按下面顺序推进，不要同时铺太多面。

### 任务 1：实现最小可训练闭环

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

### 任务 2：扩展规则边界，优先补高价值动作

建议优先级：

1. `tsumo`
2. `ankan / kakan` 的自家回合决策
3. 更完整的振听与 `ron` 过滤
4. 更严格的立直限制

原因很直接：这些动作会直接影响标签空间定义，最好在大规模训练前补稳。

### 任务 3：继续增强诊断与抽查能力

建议方向：

- 失败 case 导出
- 样本抽查 CLI
- 更细的动作级质量报告
- 大规模预处理阶段的异常分片汇总
