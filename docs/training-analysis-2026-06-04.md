# 2026-06-04 P800 single-task retrain analysis

本文是 2026-06-04 单任务复训的历史快照。当前默认 checkpoint、40w DDP 结果和 response operating point 见 [status.md](status.md)。

本轮基于 `start.sh` 重新跑了单任务 `turn_p800` 与 `response_p800` 的训练、test eval 和 response valid/test calibration。归档目录为 `runs/2026-06-04_p800_single_v1_2_safety/`。

注意：`runs/` 被 `.gitignore` 忽略，长期可提交结论以本文为准。run 目录保存了 checkpoint、日志、eval、calibration 和 `event_validation.json`。

## 1. 结论

- `turn`：可以提升为当前主候选。本轮 test `top1=71.98%`、`top3=95.72%`、`top5=98.96%`、`macro_top1=91.53%`，整体优于 05-10 improved 和 05-20 multitask `best_turn.pt`。
- `response`：不是无条件替代。raw test 比 05-10 improved 有更低 false call、更高 non-pass F1 和更高 multi-candidate top1，但 exact `chi`、`daiminkan` 和 macro 指标回退。
- `response calibration`：`best.pt` 与 `best_response.pt` 曲线一致。当时 `pass_bias=0.0 + no_yaku_call_penalty=0.4` 是最值得实战复核的新候选；当前仓库默认已在 40w DDP 后切到 06-24 `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8`。

## 2. 归档内容

```text
runs/2026-06-04_p800_single_v1_2_safety/
  run_summary.json
  analysis.md
  dataset/event_validation.json
  turn/checkpoints/
  turn/logs/
  turn/eval/turn_p800_test_eval.json
  response/checkpoints/
  response/logs/
  response/eval/response_p800_test_eval.json
  response/calibration/
```

本轮训练产物来自：

```text
outputs/checkpoints/turn_p800/
outputs/checkpoints/response_p800/
outputs/logs/turn_p800/
outputs/logs/response_p800/
outputs/reports/turn_p800_test_eval.json
outputs/reports/response_p800_test_eval.json
outputs/reports/response_p800_*calibration_*.json
outputs/reports/event_validation.json
```

## 3. Turn 对比

| 指标 | 05-06 baseline | 05-10 improved | 05-20 multitask `best_turn` | 06-04 current |
| --- | ---: | ---: | ---: | ---: |
| loss | `0.8206` | `0.8000` | `0.7731` | `0.7597` |
| top1 | `70.08%` | `70.75%` | `71.88%` | `71.98%` |
| top3 | `94.46%` | `94.93%` | `95.59%` | `95.72%` |
| top5 | `98.53%` | `98.68%` | `98.92%` | `98.96%` |
| macro top1 | `91.34%` | `90.57%` | `91.06%` | `91.53%` |
| discard top1 | `69.58%` | `70.24%` | `71.42%` | `71.49%` |
| riichi_discard top1 | `85.34%` | `87.13%` | `85.94%` | `87.89%` |
| ankan top1 | `96.02%` | `89.88%` | `92.00%` | `92.86%` |
| kakan top1 | `97.26%` | `96.27%` | `97.12%` | `97.00%` |
| non_discard top1 | `90.49%` | `91.37%` | `90.73%` | `91.99%` |

相对 05-10 improved：

- `top1/top3/top5` 分别提升 `+1.23pp / +0.79pp / +0.28pp`。
- `discard top1` 提升 `+1.24pp`。
- `riichi_discard top1` 提升 `+0.76pp`。
- `non_discard top1` 提升 `+0.62pp`。

相对 05-20 multitask `best_turn.pt`：

- `top1/top3/top5` 分别提升 `+0.10pp / +0.12pp / +0.04pp`。
- `macro top1` 提升 `+0.47pp`。
- `riichi_discard top1` 提升 `+1.95pp`。
- `non_discard top1` 提升 `+1.26pp`。
- `kakan top1` 轻微下降约 `0.11pp`。

判断：本轮 `turn_p800/best.pt` 是当前最好的 20w turn 候选。之前对 turn 少数类的担心在这一轮明显缓解，尤其是 `riichi_discard` 与 `non_discard`。

## 4. Response raw test 对比

| 指标 | 05-06 baseline | 05-10 improved | 05-20 multitask `best_response` | 06-04 current |
| --- | ---: | ---: | ---: | ---: |
| overall top1 | `98.707%` | `98.770%` | `98.728%` | `98.846%` |
| macro top1 | `89.74%` | `90.60%` | `89.85%` | `89.08%` |
| multi-candidate top1 | `86.19%` | `86.86%` | `86.41%` | `87.67%` |
| multi-candidate macro top1 | `87.42%` | `88.43%` | `87.60%` | `87.10%` |
| non-pass F1 | `71.31%` | `72.14%` | `71.41%` | `73.24%` |
| false call rate | `1.183%` | `1.105%` | `1.142%` | `1.009%` |
| missed call rate | `6.44%` | `7.28%` | `7.53%` | `8.28%` |
| false / missed calls | `359516 / 33634` | `335932 / 38033` | `347315 / 39338` | `306750 / 43268` |
| chi / pon top1 | `83.76% / 95.28%` | `81.98% / 94.75%` | `81.03% / 94.85%` | `79.68% / 94.08%` |
| daiminkan top1 | `60.76%` | `68.17%` | `64.59%` | `61.86%` |

相对 05-10 improved：

- false call 从 `1.105%` 降到 `1.009%`，少 `29182` 条。
- non-pass F1 从 `72.14%` 升到 `73.24%`。
- multi-candidate top1 从 `86.86%` 升到 `87.67%`。
- missed call 从 `7.28%` 升到 `8.28%`，多 `5235` 条。
- `chi top1` 从 `81.98%` 降到 `79.68%`。
- `daiminkan top1` 从 `68.17%` 降到 `61.86%`。

判断：本轮 response 更适合降低误叫和提升 pass/non-pass 二分类，不适合作为 exact action 全面替代。默认 serving 模型是否切换，不能只看 non-pass F1。

## 5. Response calibration

本轮 `best.pt` 和 `best_response.pt` 的 calibration 曲线一致。下面只列 test split 的关键点。

| checkpoint | pass_bias | no_yaku_penalty | false call | no-yaku false | missed call | non-pass F1 | multi top1 | chi / pon / daiminkan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 05-10 improved | `0.3` | `0.0` | `0.925%` | n/a | `9.29%` | `74.18%` | `88.40%` | `78.4 / 93.1 / 61.6` |
| 06-04 current | `0.0` | `0.0` | `1.009%` | `0.336%` | `8.28%` | `73.24%` | `87.67%` | `79.6 / 94.1 / 61.9` |
| 06-04 current | `0.0` | `0.4` | `0.915%` | `0.242%` | `9.49%` | `74.27%` | `88.45%` | `77.0 / 93.4 / 61.5` |
| 06-04 current | `0.16` | `0.0` | `0.927%` | `0.297%` | `9.21%` | `74.19%` | `88.37%` | `78.0 / 93.4 / 57.4` |
| 06-04 current | `0.16` | `0.4` | `0.841%` | `0.210%` | `10.47%` | `75.09%` | `89.06%` | `75.3 / 92.7 / 57.2` |
| 06-04 current | `0.3` | `0.4` | `0.780%` | `0.185%` | `11.36%` | `75.75%` | `89.54%` | `73.7 / 92.0 / 54.0` |

关键判断：

- `no_yaku_call_penalty=0.4` 有效。raw `pass_bias=0.0` 下，总 false call 从 `1.009%` 降到 `0.915%`，无役 false call 从 `0.336%` 降到 `0.242%`。
- 代价也明确：同一点 missed call 从 `8.28%` 升到 `9.49%`，`chi top1` 从 `79.6%` 降到 `77.0%`。
- `pass_bias=0.16 + no_yaku_call_penalty=0.4` 更保守，false call 降到 `0.841%`，但 missed call 到 `10.47%`，`chi/daiminkan` 进一步下滑。
- 不建议把 `pass_bias=0.3 + no_yaku_call_penalty=0.4` 直接作为默认；它的 missed call 已到 `11.36%`，exact `chi/daiminkan` 下降明显。

当时建议：

- 旧默认继续保留：05-10 improved `pass_bias=0.3`。
- 新候选进入实战复核：06-04 current `pass_bias=0.0 + no_yaku_call_penalty=0.4`。
- 若实战仍有明显误叫，再试 06-04 current `pass_bias=0.16 + no_yaku_call_penalty=0.4`。

当前仓库默认已在 2026-06-24 40w DDP 与 2026-06-25 扩展校准后切到 `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8`；本节保留为 06-04 历史分析。

## 6. 数据证据说明

本轮 train/eval 报告中的 P800 20w 样本计数为：

- turn train/valid/test：`83077288 / 10404270 / 10393095`
- response original train/valid/test：`247177362 / 30955527 / 30923082`
- response filtered train：`23120538`

归档目录保存了 `outputs/reports/event_validation.json`。但当前工作区的 `data/manifests/manifest_summary.json` 仍是本地旧小数据时间戳和样本数，所以没有把 `data/` 当作新数据集证据归档。短期仍引用历史数据集 ID：`tenhou_20w_2026-05-06`。

如果后续要把“重建数据”也作为正式数据版本，需要新建 `runs/_datasets/<new_dataset_id>/`，同时归档 `preprocess_summary.json`、`build_summary.json`、`manifest_summary.json` 和 split manifest。

## 7. 当时的下一步

1. `turn`：把 06-04 `turn_p800/best.pt` 作为当前主候选，先做 serving load smoke，再更新部署配置或 release note。
2. `response`：本轮当时不直接替换 05-10 improved 默认；当前仓库默认已在 40w DDP 后切到 06-24 `best_response.pt + pass_bias=2.0 + no_yaku_call_penalty=0.8`。
3. 错例抽样：按两个 response operating point 分别抽 `pass->chi`、`pass->pon`、`chi->pass`、`pon->pass`，并单独标记无役 false call。
4. 数据归档：下一轮如果确实重建数据，需要补齐新的 dataset archive，而不是只靠 `outputs/reports/event_validation.json`。
