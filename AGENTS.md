# Repository Guidelines

## Project Structure & Module Organization

This repository has moved past the planning stage. Primary documents are:

- `docs/README.md`: documentation index and recommended reading order.
- `docs/architecture-plan.md`: long-term AI, data, model, and training strategy.
- `docs/engineering-guide.md`: current implementation layout, CLI, configuration, and data contracts.
- `docs/status.md`: current implementation status, validated results, and next priorities.
- `docs/smoke-training.md`: hands-on smoke data, training, eval, and debugging workflow.
- `docs/rule-coverage.md`: current rule-engine coverage and known unsupported boundaries.

Current layout:

- `mjai/`: Python source package.
- `mjai/data/`: Tenhou XML parsing, event schemas, and dataset builders.
- `mjai/rules/`: tile representation, game state, hand-shape checks, legal actions, and replay validation. Full scoring is not implemented.
- `mjai/features/`: fixed-layout state/history and candidate-action feature encoders.
- `mjai/models/`: single-task and multitask MLP candidate rankers. There is no value head.
- `mjai/training/`: training engine, losses, class balance, and metrics.
- `mjai/eval/`: offline evaluation, distributed aggregation, calibration, and error sampling.
- `mjai/serving/` and `mjai/api/`: checkpoint inference and FastAPI routes.
- `configs/`: YAML configs for data, model, task, and environment.
- `tests/`: unit and integration tests with fixtures under `tests/fixtures/`.
- `data/`, `outputs/`, and `runs/`: local datasets, samples, logs, checkpoints, reports, and run evidence. Do not commit generated files.

## Build, Test, and Development Commands

Common commands:

```bash
python -m mjai.cli --help
python -m mjai.cli preprocess --input data/raw_xml --output data/bronze_events
python -m mjai.cli validate-events --input data/bronze_events --report outputs/reports/event_validation.json
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples --workers 32 --skip-validation
python -m mjai.cli make-manifest --input data/silver_samples --output data/manifests
python -m mjai.cli train --config configs/turn_local_smoke.yaml
python -m mjai.cli train --config configs/response_local_smoke.yaml
python -m mjai.cli eval --ckpt outputs/checkpoints/turn_local_smoke/best.pt --split valid --report outputs/checkpoints/turn_local_smoke/eval_valid.json
python -m mjai.cli inspect-sample --dataset data/silver_samples --index 0
python -m mjai.cli dump-case --game-id <game_id> --kyoku-id <kyoku_index> --step-id <step_index>
pytest
```

For local smoke data, use `tests/fixtures/tenhou` as the raw XML input. Run local smoke tests on the RTX 4070 before moving jobs to the P800 training machine.

`build-dataset` is CPU-heavy and now uses process workers when available. It also writes pre-encoded `state_features_bin` and `candidate_features_bin`, so schema changes require rebuilding `data/silver_samples` and manifests. Run `validate-events` first; after it reports `files_failed=0`, use `build-dataset --skip-validation` to avoid repeating replay validation. On P800, tune `--workers` by watching the CLI `samples_per_s` output; start around `32` and try `48` or `64` if CPU and NVMe utilization are healthy.

## Coding Style & Naming Conventions

Use Python 3.10+ style, 4-space indentation, and type hints for public interfaces. Prefer clear module names such as `legal_actions.py`, `sample_builder.py`, and `policy_net.py`. Use `snake_case` for functions, `PascalCase` for classes, and lowercase YAML names such as `turn_local_smoke.yaml`.

Keep rule logic out of model code. The rules engine must generate legal candidates before the model scores them.

## Testing Guidelines

Use `pytest`. Name tests as `test_*.py` and keep deterministic fixtures in `tests/fixtures/`. Prioritize tile conversion, red-five handling, Tenhou meld decoding, state replay, legal action generation, and sample-label validation. Any sample whose label is not in `legal_actions` must fail validation.

## Commit & Pull Request Guidelines

The repository has established history and tags. Continue the existing simple Conventional Commit style:

```text
feat: add Tenhou event parser
fix: correct red five conversion
test: add legal action fixtures
docs: update training plan
```

Pull requests should include a summary, affected modules, test results, and config changes. For training changes, include the config used and key metrics such as loss, top-k accuracy, and illegal label rate.

## Security & Configuration Tips

Do not commit raw game archives, checkpoints, local paths, credentials, or machine-specific files. Keep reusable defaults in `configs/`, and put private overrides in ignored local config files.
