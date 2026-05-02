# 雀魂四麻 AI 训练方案

## 1. 目标与范围

本项目目标是基于天凤四麻对局数据，训练一个可用于雀魂四麻场景的麻将 AI。首版至少覆盖三类核心能力：

- **弃牌决策**：自己回合从合法手牌中选择弃牌，不能弃掉不存在的牌。
- **鸣牌决策**：别人打牌后判断是否吃、碰、杠或跳过，并保证组合合法。
- **胡牌决策**：判断自摸、荣和等胡牌动作，并结合当前局面选择是否执行。

首版范围建议限定为：

- 四人麻将。
- 标准日麻规则。
- 支持赤宝牌。
- 以雀魂牌面表示为目标输出：`0m~9m`、`0p~9p`、`0s~9s`，其中 `0m/0p/0s` 为赤宝牌。
- 不优先覆盖三麻、特殊房规、娱乐模式或非标准规则。

最终目标不是只做一个离线分类器，而是形成完整链路：

```text
天凤 XML -> 事件重建 -> 状态回放 -> 合法动作生成 -> 样本构建 -> 模型训练 -> 推理校验 -> 雀魂动作输出
```

## 2. 当前环境约束

### 2.1 本地开发环境

- GPU：NVIDIA RTX 4070。
- 内存：32GB。
- 框架：`PyTorch 2.5.1+cu118`。
- 用途：小样本试运行、parser 验证、规则引擎单测、模型 smoke test、可视化调试。

### 2.2 正式训练环境

- 加速卡：8 张昆仑芯 P800。
- 显存：96GB。
- 内存：2TB。
- 硬盘：2TB。
- 框架：当前按 `PyTorch 2.5.1+cu118` 体系考虑。
- 用途：大规模数据预处理、正式训练、多实验并行、模型评估。

### 2.3 环境切换原则

本地和训练机必须使用同一套代码、同一套配置系统、同一套 CLI 入口。差异只应体现在：

- `device`：`cuda` / `xpu` / 训练环境实际设备名。
- `batch_size`。
- `num_workers`。
- `precision`。
- `distributed`。
- 数据路径。

建议所有实验通过配置文件控制，例如：

```bash
python -m mjai.cli train --task turn --config configs/turn_small.yaml
python -m mjai.cli train --task response --config configs/response_base.yaml
```

## 3. 数据来源与解析方案

### 3.1 天凤 XML 数据

当前已有约 20 万场天凤对局数据，后续可能继续扩充。XML 解析可以参考：

- `https://github.com/NegativeMjark/tenhou-log`

该项目适合作为天凤 XML 字段、事件含义、副露编码的参考资料，但不建议完全依赖其作为长期生产解析器。原因是本项目需要面向大规模训练数据构建、错误恢复、并行处理、样本版本管理和持续扩展。

### 3.2 推荐解析策略

建议自研一个面向训练的流式解析器：

- 使用 `lxml.iterparse` 或等价流式 XML parser，避免一次性加载大文件。
- 每局解析后立即进行状态回放和校验。
- 解析失败、状态不一致、规则不符合的牌谱直接记录并跳过。
- 解析结果落到中间格式，不直接喂给模型。

推荐数据分层：

```text
data/
  raw_xml/              # 原始天凤 XML
  bronze_events/        # 解析后的事件流，Parquet 格式
  silver_samples/       # 训练样本，Parquet 或 Arrow 格式
  manifests/            # 数据版本、切分、统计信息
```

## 4. 牌与动作表示

### 4.1 内部牌表示

建议内部同时维护两种牌表示：

- **34 类基础牌**：`1m~9m`、`1p~9p`、`1s~9s`、`1z~7z`。
- **赤牌标记**：单独记录 `5m/5p/5s` 中哪些是赤牌。

原因：

- 34 类表示适合计数、可见张数、手牌结构、向听数和规则判断。
- 赤牌标记适合保留雀魂动作输出所需的精确信息。
- 推理输出时再转换成雀魂牌面格式，例如 `0m`、`5m`、`E` 等内部到外部映射。

### 4.2 动作表示

不建议简单做固定类别输出，例如只预测 37 类弃牌。推荐使用“合法候选动作集合 + 排序模型”的动作表示。

每个决策点生成若干候选动作：

```text
pass
discard(tile)
riichi_discard(tile)
chi(pattern, consumed_tiles, target_tile)
pon(consumed_tiles, target_tile)
daiminkan(consumed_tiles, target_tile)
ankan(tiles)
kakan(tile)
ron
tsumo
kyuushu_kyuuhai
```

模型只对当前合法候选动作打分，最终选择最高分动作。这样可以天然避免：

- 弃掉不存在的牌。
- 吃不存在的组合。
- 对非上家吃牌。
- 不满足条件时立直。
- 振听状态下非法荣和。
- 立直后执行不允许的换牌。

## 5. 系统模块设计

建议仓库按以下模块组织：

```text
mjai/
  data/
    tenhou_parser.py       # 天凤 XML 解析
    event_schema.py        # 标准事件结构
    dataset_builder.py     # 样本构建
  rules/
    state.py               # 牌局状态
    legal_actions.py       # 合法动作生成
    scoring.py             # 胡牌与计分，后续可扩展
    validation.py          # 数据回放校验
  features/
    state_encoder.py       # 状态特征编码
    action_encoder.py      # 动作特征编码
  models/
    encoder.py             # 局面编码器
    policy.py              # 策略模型
    value.py               # 价值模型
  training/
    train_policy.py
    train_value.py
    losses.py
    metrics.py
  eval/
    offline_eval.py
    replay_eval.py
  cli.py
configs/
tests/
scripts/
```

## 6. 模型方案

### 6.1 总体推荐

推荐采用 **Hybrid Transformer Candidate Ranker**：

- 用结构化特征描述当前局面。
- 用事件序列 Transformer 捕捉牌局历史。
- 用动作编码器描述每个候选动作。
- 对每个合法候选动作输出一个分数。
- 在合法候选集合内做 softmax 和交叉熵训练。

该方案比纯 CNN 或固定动作分类更适合麻将，因为麻将动作空间是动态的，并且强依赖合法性约束。

### 6.2 输入特征

局面特征建议包含：

- 自己手牌计数。
- 赤牌持有情况。
- 四家副露。
- 四家弃牌河。
- 宝牌指示牌。
- 场风、自风、局数、本场、立直棒。
- 四家分数。
- 剩余巡目。
- 可见牌计数。
- 当前玩家是否立直。
- 当前玩家是否一发巡。
- 是否副露、是否门清。
- 是否振听。

事件序列特征建议包含最近 `128~256` 个事件：

- 玩家相对座次。
- 动作类型。
- 牌。
- 是否摸切。
- 是否立直。
- 是否副露。

候选动作特征建议包含：

- 动作类型。
- 目标牌。
- 消耗手牌。
- 来源玩家。
- 是否包含赤牌。
- 动作后是否副露。
- 动作后手牌张数变化。

### 6.3 推荐模型结构

```text
State Features -> MLP / Light CNN --------\
                                           -> Shared Mahjong Encoder -> Task Head -> scores over legal actions
Event Tokens -> Transformer Encoder ------/
Action Features -> Action Encoder --------/
```

首版规模建议：

- `turn_small`：2~4 层 Transformer，hidden size 256。
- `response_small`：2~4 层 Transformer，hidden size 256。
- `turn_base`：6~8 层 Transformer，hidden size 512。
- `response_base`：6~8 层 Transformer，hidden size 512。

## 7. 模型拆分

建议按三个任务拆分系统逻辑，但工程实现上推荐使用**共享编码器 + 多头候选动作排序模型**，而不是三个完全独立模型，也不是三个简单固定分类头。

推荐结构：

```text
Shared Mahjong Encoder
  -> turn_head
  -> response_head
  -> value_head
```

其中：

- `turn_head` 只处理自己回合的合法候选动作集合。
- `response_head` 只处理别人打牌后的合法候选动作集合。
- `value_head` 预测局面价值，可作为辅助训练或后续 rerank 模块。

不推荐“弃牌头 / 鸣牌头 / 胡牌头”这种简单并列分类头，原因是很多动作天然处于同一个竞争决策面。例如别人打出一张牌时，`pass / chi / pon / daiminkan / ron` 必须统一排序，而不是由多个头分别判断后再人工仲裁。

### 7.1 `turn_policy`

对应 `turn_head`，处理自己回合动作：

- 普通弃牌。
- 立直弃牌。
- 暗杠。
- 加杠。
- 自摸。
- 九种九牌。

### 7.2 `response_policy`

对应 `response_head`，处理别人打牌后的响应：

- 跳过。
- 吃。
- 碰。
- 明杠。
- 荣和。

该模型要重点处理类别不平衡，因为大多数响应点都是 `pass`。

### 7.3 `value_model`

对应 `value_head`，预测局面价值：

- 期望点数变化。
- 期望顺位变化。
- 或最终排名分布。

首版可以不参与线上决策，但建议一开始保留标签和训练入口。后续可用于：

- 多动作 rerank。
- 防守强度判断。
- 离线强化学习。
- 局面分析。

## 8. 训练路线

### 8.1 Phase 0：数据与规则验证

目标是先保证数据链路正确，而不是追求模型精度。

任务：

- 完成天凤 XML 解析。
- 完成牌局状态回放。
- 完成合法动作生成。
- 完成训练样本构建。
- 建立单元测试。
- 抽样回放人工核对。

验收指标：

- XML 解析成功率可统计。
- 标签动作必须属于合法动作集合。
- 非法标签率必须为 `0`，否则视为 parser 或规则引擎 bug。

### 8.2 Phase 1：Baseline 监督学习

目标是快速得到可训练、可评估的首版模型。

任务：

- 训练一个轻量共享编码器，并优先打开 `turn_head` 路径。
- 训练一个轻量共享编码器，并优先打开 `response_head` 路径。
- 使用小样本在本地 4070 上完成 smoke test。
- 验证 loss 能下降、top-k 指标合理、没有非法输出。

推荐命令：

```bash
python -m mjai.cli train --task turn --config configs/turn_small.yaml
python -m mjai.cli train --task response --config configs/response_small.yaml
```

### 8.3 Phase 2：正式监督训练

目标是使用全量 20 万场数据训练正式模型。

任务：

- 在训练机上跑全量数据预处理。
- 先分别训练 `turn_head` 和 `response_head`。
- 再进行一次共享编码器的多任务联合微调。
- 开启混合精度。
- 根据 P800 环境实际支持情况开启分布式训练。
- 记录所有实验配置、数据版本、checkpoint 和指标。

推荐命令：

```bash
python -m mjai.cli train --task turn --config configs/turn_base.yaml --distributed
python -m mjai.cli train --task response --config configs/response_base.yaml --distributed
```

### 8.4 Phase 3：高质量数据微调

目标是提升上限。

建议按以下维度筛高质量数据：

- 高段位。
- 高 rate。
- 特上 / 凤凰 / 等价高水平房间。
- 排除断线、异常对局、规则不一致对局。

训练方式：

- 先用全量数据训练基础模型或多头共享模型。
- 再用高质量子集低学习率微调。
- 也可以在全量训练中对高质量样本加权。

### 8.5 Phase 4：价值模型与离线强化学习

监督学习稳定后，再考虑价值模型和离线 RL。

可选路线：

- 训练 `value_model` 预测最终顺位、点数变化或局收益。
- 使用 value model 对策略模型 top-k 动作 rerank。
- 尝试 Advantage Weighted Regression、IQL 或 CQL-lite 等离线强化学习方法。

不建议一开始直接做大规模自博弈。原因是模拟器、计分、规则边界、训练稳定性都会显著增加工程复杂度。

## 9. 数据处理性能方案

### 9.1 Windows 本地

本地主要用于：

- 小批量 XML 解析。
- 规则单测。
- 训练样本抽查。
- 小模型训练。

建议避免在 Windows 上跑全量 20 万局预处理，因为：

- 文件系统小文件吞吐通常较弱。
- 多进程 DataLoader 与路径兼容更容易出问题。
- 调试可以，生产不划算。

### 9.2 Linux 训练环境

训练环境用于：

- 全量 XML 解析。
- Parquet 样本生成。
- 多进程 DataLoader。
- 大模型训练。

建议：

- XML 解析阶段按文件分片并行。
- 中间结果写 Parquet，开启 `zstd` 压缩。
- 训练样本按 shard 存储，例如每个 shard `256MB~1GB`。
- 避免一个样本一个文件。
- 保存数据 manifest，记录样本数、规则、过滤条件、切分版本。

### 9.3 数据切分

必须按 `game_id` 切分：

```text
train: 90%
valid: 5%
test: 5%
```

禁止按样本随机切分。否则同一场对局的不同决策点会同时进入训练集和验证集，导致验证指标虚高。

## 10. 合法性校验

### 10.1 解析校验

每局解析后检查：

- 136 张牌守恒。
- 初始手牌数量正确。
- 摸牌、弃牌、副露后手牌数量正确。
- 弃牌必须来自当前手牌。
- 副露组合必须可由当前手牌和目标牌组成。
- 宝牌指示牌、立直、本场、分数变化可追踪。

### 10.2 样本校验

每个样本检查：

- `label_action in legal_actions`。
- `legal_actions` 不为空。
- 自己回合不能出现别人响应动作。
- 响应回合不能出现自己摸牌后的弃牌动作。
- 赤牌组合不能丢失。

### 10.3 推理校验

推理时严格流程：

```text
当前状态 -> 规则引擎生成合法动作 -> 模型打分 -> 选择最高分合法动作 -> 输出雀魂动作
```

模型永远不能直接绕过规则引擎输出动作。

异常回退策略：

- 响应动作异常：默认 `pass`。
- 自己回合异常：从合法弃牌中选择启发式安全牌或固定 fallback。

## 11. 评估指标

### 11.1 监督学习指标

- `loss`。
- `top1 accuracy`。
- `top3 accuracy`。
- `top5 accuracy`。
- 按动作类型统计 accuracy。
- `illegal_label_rate`。
- `illegal_output_rate`。

### 11.2 响应模型专项指标

由于 `pass` 占比极高，必须单独看：

- `chi precision / recall`。
- `pon precision / recall`。
- `kan precision / recall`。
- `ron precision / recall`。
- `pass false negative`。
- `call rate` 与真实玩家 call rate 的差异。

### 11.3 对局风格指标

后续评估需要看：

- 和了率。
- 放铳率。
- 副露率。
- 立直率。
- 平均打点。
- 平均失点。
- 一位率、二位率、三位率、四位率。
- 平均顺位。

## 12. 可视化要求

建议使用 TensorBoard，或后续接入 Weights & Biases / MLflow。考虑本地与训练环境切换，首版优先使用 TensorBoard，依赖少、迁移简单。

必须记录：

- train / valid loss。
- top-k accuracy。
- 各动作类型 precision / recall。
- 学习率。
- batch time。
- data loading time。
- GPU / XPU 显存占用。
- 样本吞吐。
- 梯度范数。

建议额外实现固定牌谱可视化：

- 选取若干固定验证牌局。
- 每轮输出模型 top-k 动作。
- 对比真实动作。
- 输出错误原因分类，例如非法性、过度进攻、过度防守、错过荣和、错误副露。

## 13. 训练入口设计

建议所有流程统一到一个 CLI：

```bash
# 解析 XML
python -m mjai.cli preprocess --input data/raw_xml --output data/bronze_events

# 构建训练样本
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples

# 训练自己回合模型
python -m mjai.cli train --task turn --config configs/turn_small.yaml

# 训练响应模型
python -m mjai.cli train --task response --config configs/response_small.yaml

# 训练价值模型
python -m mjai.cli train --task value --config configs/value_small.yaml

# 离线评估
python -m mjai.cli eval --task turn --ckpt outputs/turn/best.pt

# 回放评估
python -m mjai.cli replay-eval --ckpt outputs/turn/best.pt --games data/test_games.txt
```

配置文件示例：

```yaml
seed: 20260423
device: cuda
precision: bf16
batch_size: 512
num_workers: 8
model:
  hidden_size: 256
  num_layers: 4
  num_heads: 8
optimizer:
  name: adamw
  lr: 0.0003
  weight_decay: 0.01
data:
  train_manifest: data/manifests/train.jsonl
  valid_manifest: data/manifests/valid.jsonl
logging:
  backend: tensorboard
  log_dir: outputs/logs
```

训练机只需改配置或覆盖参数：

```bash
python -m mjai.cli train --task turn --config configs/turn_base.yaml --device xpu --distributed
```

## 14. 测试计划

### 14.1 单元测试

必须覆盖：

- 牌表示转换。
- 赤牌转换。
- 天凤副露编码解析。
- 手牌增减。
- 合法弃牌。
- 合法吃牌。
- 合法碰牌。
- 合法杠。
- 荣和与振听。
- 立直条件。

### 14.2 回放测试

从原始 XML 中抽样：

- 低复杂度普通局。
- 多次副露局。
- 多次立直局。
- 包含杠和新宝牌局。
- 包含流局局。
- 包含多家荣和局。

每个样例都要保证从 `INIT` 到 `AGARI/RYUUKYOKU` 状态一致。

### 14.3 训练测试

小数据 smoke test：

- 100 局。
- 1000 局。
- 1 万局。

每个规模检查：

- 是否能完整构建样本。
- 是否能跑完一个 epoch。
- loss 是否下降。
- 是否出现 NaN。
- 是否有非法标签。

## 15. 主要风险与对策

### 15.1 天凤到雀魂分布差异

风险：

- 天凤玩家行为、段位体系、规则细节和雀魂不完全一致。

对策：

- 首版用天凤训练基础能力。
- 后续收集雀魂对局做验证和微调。
- 评估时单独关注副露率、立直率、放铳率等风格指标。

### 15.2 响应模型学成永远 pass

风险：

- `pass` 样本远多于吃碰杠荣。

对策：

- 训练时使用类别权重。
- 对 pass 做下采样或 hard negative sampling。
- 指标不能只看总体 accuracy，必须看各动作 recall。

### 15.3 规则引擎错误污染训练

风险：

- 合法动作生成错了，模型会被迫学习错误标签。

对策：

- parser、state、legal_actions 独立测试。
- 标签不在合法集合内时直接报错。
- 固定样例回放人工核验。

### 15.4 磁盘与 IO 成为瓶颈

风险：

- 20 万局及后续扩展会导致 XML、小文件、样本 shard、checkpoint 占用大量空间。

对策：

- Parquet/Arrow shard 化。
- 压缩使用 `zstd`。
- 保留数据 manifest。
- 定期清理中间实验 checkpoint。

### 15.5 P800 上 PyTorch 兼容性

风险：

- P800 的 PyTorch 运行方式、分布式后端、混合精度支持可能与 CUDA GPU 不完全一致。

对策：

- 把设备相关逻辑集中封装在 `device_utils.py`。
- 本地先保证 `cuda` 路径。
- 训练机单卡 smoke test 通过后再开多卡。
- 混合精度、分布式、DataLoader workers 分阶段打开。

## 16. 建议里程碑

### Milestone 1：工程骨架与规则核心

- 建立项目目录。
- 实现牌表示。
- 实现基础状态机。
- 实现合法弃牌、吃、碰、杠、胡的接口。
- 建立单元测试。

### Milestone 2：天凤数据解析与样本生成

- 实现 XML parser。
- 完成事件流 schema。
- 完成回放校验。
- 生成首批训练样本。
- 输出数据统计报告。

### Milestone 3：本地小模型训练

- 实现共享编码器和 `turn_head`。
- 实现共享编码器和 `response_head`。
- 在本地 4070 上完成 1000~10000 局 smoke test。
- 建立 TensorBoard 日志。

### Milestone 4：训练机全量训练

- 完成训练机环境适配。
- 跑通单卡训练。
- 跑通多卡训练。
- 用全量 20 万局训练 base 模型。

### Milestone 5：评估与改进

- 建立离线评估报告。
- 建立固定牌谱回放评估。
- 分析典型错例。
- 做高质量数据微调。
- 训练 value model。

## 17. 当前推荐结论

当前最稳妥的方案是：

- 使用 `PyTorch 2.5.1+cu118` 作为统一训练框架。
- 数据处理使用 `Parquet/Arrow` 中间格式。
- 天凤 XML 解析参考 `tenhou-log`，但实现自己的训练级 parser。
- 以规则引擎作为训练和推理的共同前置模块。
- 模型采用 `Hybrid Transformer Candidate Ranker`。
- 工程实现采用共享编码器多头结构：`turn_head + response_head + value_head`。
- 系统逻辑仍按 `turn_policy`、`response_policy`、`value_model` 三个任务管理。
- 首版走监督学习，稳定后再做高质量微调、价值模型和离线强化学习。

这一路线优先保证工程闭环、动作合法性和环境可迁移性，再逐步追求更高牌力。
