"""Run six independent single-device evaluations on a Linux P800 node.

Usage: python scripts/eval_p800_parallel.py [--dry-run] [--cards 0,1,2,3,4,5]
Use nohup or tmux to keep the supervisor alive after disconnecting.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def run_worker(card: int, cli_args: list[str]) -> None:
    """Bind the current device before the CLI creates models or tensors."""
    import runpy
    import pyarrow
    import torch

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or card >= torch.cuda.device_count():
        raise RuntimeError(f"Requested visible device {card} is unavailable")
    torch.cuda.set_device(card)
    print(f"PID={os.getpid()} device={torch.cuda.current_device()} "
          f"name={torch.cuda.get_device_name(card)}", flush=True)
    sys.argv = ["mjai.cli", *cli_args]
    runpy.run_module("mjai.cli", run_name="__main__")


def main() -> int:
    """Validate inputs, launch every job, and record completion status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", default="0,1,2,3,4,5")
    parser.add_argument("--config", default="configs/eval_full_state_p800_parallel.yaml")
    parser.add_argument("--ckpt-dir", default="outputs/checkpoints/multitask_full_state_v2_ddp")
    parser.add_argument("--run-dir", help="New directory for logs and reports; must not exist")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing or launching")
    args = parser.parse_args()
    cards = [int(value) for value in args.cards.split(",")]
    if len(cards) != 6 or len(set(cards)) != 6 or min(cards) < 0:
        parser.error("--cards requires six distinct nonnegative visible device indices")
    os.chdir(ROOT)
    run_dir = Path(args.run_dir or (
        "outputs/reports/parallel_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )).resolve()
    config_path = Path(args.config).resolve()
    jobs = []
    for split in ("valid", "test"):
        for kind in ("turn", "response", "calibration"):
            name = f"{kind}_{split}"
            head = "turn" if kind == "turn" else "response"
            command = ["calibrate-response" if kind == "calibration" else "eval",
                       "--ckpt", str(Path(args.ckpt_dir) / f"best_{head}.pt"),
                       "--split", split, "--config", str(config_path),
                       "--report", str(run_dir / f"{name}.json")]
            if kind != "calibration":
                command += ["--task", kind]
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--worker", str(cards[len(jobs)]), *command]
            jobs.append((name, command))
            print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if os.name != "posix":
        parser.error("Execution requires the Linux training node; use --dry-run locally")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Run with python, outside torchrun")
    from mjai.utils.config import load_resolved_yaml
    config = load_resolved_yaml(config_path)
    if config.get("eval", {}).get("distributed", False):
        parser.error("eval.distributed must be false")
    if config.get("train", {}).get("device") != "cuda":
        parser.error("train.device must be cuda for explicit device binding")
    required = [Path(args.ckpt_dir) / f"best_{head}.pt" for head in ("turn", "response")]
    required += [Path(config["data"][f"{split}_manifest"]) for split in ("valid", "test")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    import torch
    if not torch.cuda.is_available() or max(cards) >= torch.cuda.device_count():
        parser.error(f"Requested cards {cards}; visible count={torch.cuda.device_count()}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    env["ARROW_NUM_THREADS"] = "1"
    processes = []
    records = []

    def interrupted(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for card, (name, command) in zip(cards, jobs):
            with (run_dir / f"{name}.log").open("wb") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            records.append(dict(name=name, card=card, pid=process.pid, command=command, exit_code=None))
            print(f"Started {name}: card={card} PID={process.pid}", flush=True)
        pending = set(range(len(processes)))
        while pending:
            for index in list(pending):
                code = processes[index].poll()
                if code is not None:
                    records[index]["exit_code"] = code
                    pending.remove(index)
                    print(f"Finished {records[index]['name']}: exit={code}", flush=True)
            (run_dir / "status.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            if pending:
                time.sleep(2)
    finally:
        # Stop only process groups launched by this supervisor, including loaders.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    print(f"Reports and logs: {run_dir}", flush=True)
    return int(any(record["exit_code"] != 0 for record in records))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        run_worker(int(sys.argv[2]), sys.argv[3:])
    else:
        raise SystemExit(main())
