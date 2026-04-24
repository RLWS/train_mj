# Smoke Train 手动验证教程

本文档说明如何在当前仓库里手动验证最小训练闭环，包括：

- 生成 smoke 用数据集
- 运行 `turn` smoke train
- 运行 `response` smoke train
- 查看训练输出
- 修改数据集路径

适用目录：

- 仓库根目录 [train_mj](F:\Code\AI\train_study\train_mj)

## 1. 前置条件

确保当前终端位于仓库根目录：

```powershell
cd F:\Code\AI\train_study\train_mj
```

当前训练命令依赖本地可用的 `torch` 环境。

## 2. 先准备 smoke 数据集

最简单的方式是直接使用仓库自带的天凤夹具：

```powershell
python -m mjai.cli preprocess --input tests/fixtures/tenhou --output data/bronze_events
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
```

执行完成后会得到：

- 事件分片目录：[data/bronze_events](F:\Code\AI\train_study\train_mj\data\bronze_events)
- 样本分片目录：[data/silver_samples](F:\Code\AI\train_study\train_mj\data\silver_samples)
- 清单目录：[data/manifests](F:\Code\AI\train_study\train_mj\data\manifests)

其中最关键的清单文件是：

- [train.jsonl](F:\Code\AI\train_study\train_mj\data\manifests\train.jsonl)
- [valid.jsonl](F:\Code\AI\train_study\train_mj\data\manifests\valid.jsonl)
- [test.jsonl](F:\Code\AI\train_study\train_mj\data\manifests\test.jsonl)

## 3. 运行 turn smoke train

命令：

```powershell
python -m mjai.cli train --config configs/turn_local_smoke.yaml
```

使用的配置文件：

- [turn_local_smoke.yaml](F:\Code\AI\train_study\train_mj\configs\turn_local_smoke.yaml)

这个配置默认读取：

- `data/manifests/train.jsonl`
- `data/manifests/valid.jsonl`

## 4. 运行 response smoke train

命令：

```powershell
python -m mjai.cli train --config configs/response_local_smoke.yaml
```

使用的配置文件：

- [response_local_smoke.yaml](F:\Code\AI\train_study\train_mj\configs\response_local_smoke.yaml)

这个配置同样默认读取：

- `data/manifests/train.jsonl`
- `data/manifests/valid.jsonl`

## 5. 数据集路径在哪里配置

训练时读取的数据路径都在配置文件里，关键字段是：

```yaml
data:
  train_manifest: data/manifests/train.jsonl
  valid_manifest: data/manifests/valid.jsonl
  candidate_limit: 64
```

也就是说：

- 如果你用默认 smoke 数据，就不需要改
- 如果你要换成自己的数据集，只需要把 `train_manifest` 和 `valid_manifest` 改成你自己的 manifest 路径

对应配置文件位置：

- [turn_local_smoke.yaml](F:\Code\AI\train_study\train_mj\configs\turn_local_smoke.yaml)
- [response_local_smoke.yaml](F:\Code\AI\train_study\train_mj\configs\response_local_smoke.yaml)

## 6. 训练输出在哪里看

### turn

默认输出目录：

- checkpoint：[turn_local_smoke](F:\Code\AI\train_study\train_mj\outputs\checkpoints\turn_local_smoke)
- log：[turn_local_smoke](F:\Code\AI\train_study\train_mj\outputs\logs\turn_local_smoke)

### response

默认输出目录：

- checkpoint：[response_local_smoke](F:\Code\AI\train_study\train_mj\outputs\checkpoints\response_local_smoke)
- log：[response_local_smoke](F:\Code\AI\train_study\train_mj\outputs\logs\response_local_smoke)

重点看这些文件：

- `config.resolved.yaml`
- `train_metrics.json`
- `best.pt`
- `final.pt`
- `epoch_metrics.jsonl`

## 7. 如何判断 smoke train 成功

成功的最低标准是：

- 训练命令正常结束，没有异常退出
- 输出目录里生成了：
  - `config.resolved.yaml`
  - `train_metrics.json`
  - `best.pt`
  - `final.pt`
- `train_metrics.json` 中存在：
  - `epoch_summaries`
  - `train_samples`
  - `valid_samples`

## 8. 一套最短命令清单

如果你只是想从零跑一遍最短流程，可以直接执行：

```powershell
python -m mjai.cli preprocess --input tests/fixtures/tenhou --output data/bronze_events
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
python -m mjai.cli train --config configs/turn_local_smoke.yaml
python -m mjai.cli train --config configs/response_local_smoke.yaml
```

## 9. 如果你要换成自己的数据

推荐流程：

1. 先自己准备原始 XML 目录
2. 运行 `preprocess`
3. 运行 `build-dataset`
4. 运行 `make-manifest`
5. 修改训练配置里的 `train_manifest` / `valid_manifest`
6. 再运行 `train`

示例：

```powershell
python -m mjai.cli preprocess --input D:\your_xml_dir --output D:\your_data\bronze_events
python -m mjai.cli build-dataset --input D:\your_data\bronze_events --output D:\your_data\silver_samples
python -m mjai.cli make-manifest --input D:\your_data\silver_samples --output D:\your_data\manifests
python -m mjai.cli train --config configs/turn_local_smoke.yaml
```

然后把配置里的：

```yaml
data:
  train_manifest: D:/your_data/manifests/train.jsonl
  valid_manifest: D:/your_data/manifests/valid.jsonl
```

改成你自己的路径即可。
