# P800 六任务并行评估

在 Linux 训练容器的项目根目录执行。同步以下两个文件即可，模型结构和训练参数继续从 checkpoint 读取：

- `scripts/eval_p800_parallel.py`
- `configs/eval_full_state_p800_parallel.yaml`

先查看计划（不启动任务、不写报告）：

```bash
python scripts/eval_p800_parallel.py --dry-run
```

后台运行：

```bash
mkdir -p outputs/logs
nohup python -u scripts/eval_p800_parallel.py > outputs/logs/parallel_eval_launcher.log 2>&1 < /dev/null &
tail -f outputs/logs/parallel_eval_launcher.log
```

默认使用当前容器可见卡编号 0–5。依次执行 turn valid、response valid、calibration valid、turn test、response test、calibration test，六个进程同时启动。卡 6、7 保留。可以通过 `--cards 2,3,4,5,6,7` 改用其他六张卡；启动前确认选择的卡没有其他评估任务占用。

每次自动创建 `outputs/reports/parallel_eval_<时间戳>/`，包含六份 JSON、六份日志、配置副本和 `status.json`。监督进程等待全部任务结束；任何任务失败，最终返回非零退出码。不会覆盖此前报告。向监督进程发送 SIGTERM 会清理它启动的任务和加载进程。

默认每个任务 12 个加载进程，共 72 个；CPU 数学库与 Arrow 线程限制为 1，避免超线程被多层并行占满。turn batch 为 16384、response 为 32768，预取为每个 worker 两批。适用于 96 核、1.5 TB 内存、每卡 96 GB 的节点，属于待实机测量的起始配置，不代表已测得最优吞吐。

容器共享内存同样重要：运行 `df -h /dev/shm` 检查，默认 64 MB 不适用。加载队列可能占用数十至数百 GB，建议容器提供至少 256 GB 共享内存或使用节点已有的 host IPC 配置。若出现 DataLoader bus error，先检查共享内存；若显存不足，将两个 batch 调回 8192/16384；若磁盘满负荷但卡利用率低，先将 workers 降为 8，避免继续增加读盘竞争。

两个 calibration 都会扫描默认阈值网格。最终阈值只能根据 valid 的结果选择，再读取 test 对应点的指标；不要使用 test 报告推荐值重新调参。

本地已检查 Python 语法、配置解析和六命令生成；尚未在 P800 节点实测吞吐或峰值显存。
