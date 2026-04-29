# 项目进度状态

更新时间：`2026-04-29`

## 1. 总体判断

当前项目已经明显超过“纯规划阶段”，可以视为：

- `Phase 0` 第一版已完成：规则与数据的最小闭环已经跑通。
- `Phase 1` 已启动：`turn/response` 单任务 baseline 都已能本地 smoke train，并可做第一版离线评估。

换句话说，仓库当前已经从“只能产样本”进入“能产样本并训练最小 baseline”，但训练体系还远没完整。

## 2. 已验证结果

以下结果已经在仓库内实际验证：

- `pytest` 全量通过，当前结果为 `84 passed`。
- `python -m mjai.cli preprocess --input tests/fixtures/tenhou --output <tmp>` 可运行。
- 对 `tests/fixtures/tenhou` 中的 `20` 个 XML 夹具实测，生成 `6281` 条事件并写入 `bronze_events`。
- `python -m mjai.cli validate-events --input <tmp> --report <tmp>` 可运行。
- 同一批事件分片实测回放结果为：
  - `files=20`
  - `valid=20`
  - `failed=0`
  - `total_events=6281`
  - 可输出事件分布、失败原因聚合、总局数/弃牌数/副露数摘要
- `python -m mjai.cli build-dataset --input <tmp> --output <tmp>` 可运行，且支持 `--workers N` 按事件分片并行构建。
- `python -m mjai.cli build-dataset --input <tmp> --output <tmp> --workers 2` 已在 Windows fixture smoke 中验证通过。
- 对 `tests/fixtures/tenhou` 中的 `20` 个 XML 夹具实测，生成：
  - `2996` 条 `turn` 样本
  - `8910` 条 `response` 样本
  - 可输出 `turn/response` 标签动作分布，其中当前真实夹具的 `turn` 已覆盖 `discard / riichi_discard / tsumo / ankan / kyuushu_kyuuhai`，构建链路本身已支持 `kakan`
- `python -m mjai.cli make-manifest --input <tmp> --output <tmp>` 可运行。
- 同一批夹具可生成 manifest，实测分桶结果为：
  - `train_games=18`
  - `valid_games=1`
  - `test_games=1`
- `python -m mjai.cli train --config <tmp>` 可运行。
- 以单局清单裁剪出的最小训练集实测，`turn` 与 `response` 任务都可完成 `1` 个 epoch 的 smoke train，并落盘：
  - `config.resolved.yaml`
  - `train_metrics.json`
  - `best.pt`
  - `final.pt`
- `python -m mjai.cli eval --ckpt <tmp> --split valid --report <tmp>` 可运行。
- 对最小 `turn` 与 `response` smoke checkpoint 实测，可输出：
  - `loss`
  - `top1 / top3 / top5 accuracy`
  - 按动作类别聚合的样本数与 `top-k` 指标
  - 按候选动作数量分桶的 `top-k` 指标
  - 首批错例样本的 label、top-k 预测、legal actions，以及可直接执行的 `inspect-sample` / `dump-case` 跳转命令
  - JSON 评估报告
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
  - `build-dataset` 已支持 `--workers N` 并行处理事件分片，当前实现使用线程池以避开 Windows 进程池启动和权限问题。
  - Parquet 写出与 manifest 生成已实现。
  - 事件分布、动作分布、split 样本统计和 warning 已实现第一版。
- `configs/`
  - 数据、任务、模型、训练配置目录已建立。
- `mjai/features/`
  - 已有最小 `turn/response` 状态特征与动作特征编码。
- `mjai/models/`
  - 已有最小共享编码器与 `turn/response` 候选打分头。
- `mjai/training/`
  - 已有最小训练循环、加权 loss、accuracy 指标与 checkpoint 落盘。
- `mjai/eval/`
  - 已有第一版离线评估入口，可基于 checkpoint 复用训练配置做 split 级评估与 JSON 报告导出。
  - 已支持候选规模分桶与首批错例导出。
- `tests/`
  - 规则、解析、样本构建、CLI、配置和数据写出都有测试覆盖。
  - `turn/response` smoke train、offline eval、`inspect-sample` 与 `dump-case` 都有回归测试。

### 部分完成

- `mjai/cli.py`
  - 子命令框架完整。
  - `preprocess`、`validate-events`、`build-dataset`、`make-manifest`、`train`、`eval`、`inspect-sample`、`dump-case` 已可用。
  - `replay-eval`、`infer-on-state` 仍是占位实现。
- 数据链路
  - 已经落地 `raw_xml -> bronze_events -> silver_samples` 两阶段流水线。
- 第一版诊断输出已经落地，并已支持从评估错例跳转到样本与事件上下文，但还缺少更细的失败样本聚合和固定 case 管理。

### 尚未开始或基本未开始

- `mjai/serving/`

这些目录目前仍只有包骨架或说明性占位，还不能支持推理闭环。

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

- 更完整的和牌机会放弃语义与残余振听边界
- 多家荣和等边界情况
- 更完整的立直限制与流局处理

### 4.3 训练与评估链路已打通最小版本，但覆盖面还很窄

当前已经有：

- `turn` 状态特征编码
- `response` 状态特征编码
- `turn/response` 候选动作编码
- 最小共享编码器和 `turn/response` 候选排序头
- `turn/response` 训练循环
- `loss / accuracy`
- `turn/response` 离线 `top-k` 评估与按类别聚合报告
- checkpoint 与训练指标落盘

下一阶段优先补：

- 更大数据量的正式训练配置
- 更长训练轮数与更可靠的 eval report 留存
- 训练过程中的动作分布、候选规模和非法标签质量闸门
- 根据错例聚合决定是否回补规则

## 5. 阶段结论

如果按原始路线图划分，当前可以认为：

- `Task 1` 项目骨架初始化：已完成
- `Task 2` 牌表示与基础规则：已完成
- `Task 3` 天凤解析与事件 schema：已完成
- `Task 4` 回放与合法动作生成：已完成第一版
- `Task 5` 样本构建：已完成第一版
- `preprocess / validate-events`：已完成第一版
- `Task 6` baseline 模型与训练引擎：已完成 `turn/response` 第一版
- `Task 7` 离线评估：已完成第一版
- `Task 8` 多任务训练：未开始
- `Task 9` 训练机适配：只完成配置骨架，未验证训练链路

## 6. 建议的下一步任务

建议按下面顺序推进：训练先行，规则边界按证据回补。

### 任务 1：扩大数据构建与质量闸门

目标：

- 在更多真实牌谱上跑完整数据链路，确认当前规则集对训练数据的污染率是否可接受。

建议范围：

1. 扩大输入牌谱规模，跑通 `preprocess / validate-events / build-dataset --workers N / make-manifest`
2. 汇总构建失败率、非法标签率、动作分布、候选规模分布和 split 样本数
3. 对失败样本只保留轻量定位信息，依赖 `inspect-sample` 和 `dump-case` 回溯上下文

最低验收标准：

- 更大数据集能稳定产出 `turn/response` manifest。
- 非法标签率和构建失败率有明确统计。
- 发现的高频失败能定位到具体 `game_id / kyoku_index / step_index`。

### 任务 2：深入训练当前 `turn/response` baseline

建议范围：

1. 增大训练数据量、epoch、hidden size 和 batch 配置
2. 分别训练 `turn` 与 `response`，固定输出 `train_metrics.json`、checkpoint 和 eval report
3. 对比不同配置的 loss、top-k accuracy、per-kind 指标和候选规模分桶
4. 记录可复现实验配置，避免只看单次 smoke 结果

### 任务 3：规则边界按训练反馈回补

建议方向：

- 更完整的和牌机会放弃语义与残余振听边界
- 其他途中流局边界
- 更严格的立直限制
- 多家荣和等边界情况

触发条件：

- 某类规则缺口在大规模构建中导致高频失败或非法标签。
- 某类规则缺口在 eval 错例中形成稳定、可复现的主要误差来源。
