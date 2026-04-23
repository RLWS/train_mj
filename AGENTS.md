# Repository Guidelines

## Project Structure & Module Organization

This repository is in planning stage. Primary documents are:

- `MAHJONG_AI_PLAN.md`: overall AI, data, model, and training strategy.
- `MAHJONG_AI_ENGINEERING.md`: proposed implementation layout, CLI design, configuration, and first tasks.

When implementation starts, follow the planned layout:

- `mjai/`: Python source package.
- `mjai/data/`: Tenhou XML parsing, event schemas, and dataset builders.
- `mjai/rules/`: tile representation, game state, legal actions, validation, and scoring.
- `mjai/features/`: state, history, and action encoders.
- `mjai/models/`: shared encoder plus `turn_head`, `response_head`, `value_head`.
- `mjai/training/`: losses, metrics, optimizers.
- `configs/`: YAML configs for data, model, task, and environment.
- `tests/`: unit and integration tests with fixtures under `tests/fixtures/`.
- `data/` and `outputs/`: local datasets, samples, logs, checkpoints, reports. Do not commit generated files.

## Build, Test, and Development Commands

Planned commands:

```bash
python -m mjai.cli --help
python -m mjai.cli preprocess --input data/raw_xml --output data/bronze_events
python -m mjai.cli build-dataset --input data/bronze_events --output data/silver_samples
python -m mjai.cli train --config configs/turn_small.yaml
python -m mjai.cli eval --ckpt outputs/checkpoints/turn_small/best.pt --split valid
pytest
```

Run local smoke tests on the RTX 4070 before moving jobs to the P800 training machine.

## Coding Style & Naming Conventions

Use Python 3.10+ style, 4-space indentation, and type hints for public interfaces. Prefer clear module names such as `legal_actions.py`, `sample_builder.py`, and `policy_net.py`. Use `snake_case` for functions, `PascalCase` for classes, and lowercase YAML names such as `turn_small.yaml`.

Keep rule logic out of model code. The rules engine must generate legal candidates before the model scores them.

## Testing Guidelines

Use `pytest`. Name tests as `test_*.py` and keep deterministic fixtures in `tests/fixtures/`. Prioritize tile conversion, red-five handling, Tenhou meld decoding, state replay, legal action generation, and sample-label validation. Any sample whose label is not in `legal_actions` must fail validation.

## Commit & Pull Request Guidelines

There is no git history yet, so use a simple convention going forward:

```text
feat: add Tenhou event parser
fix: correct red five conversion
test: add legal action fixtures
docs: update training plan
```

Pull requests should include a summary, affected modules, test results, and config changes. For training changes, include the config used and key metrics such as loss, top-k accuracy, and illegal label rate.

## Security & Configuration Tips

Do not commit raw game archives, checkpoints, local paths, credentials, or machine-specific files. Keep reusable defaults in `configs/`, and put private overrides in ignored local config files.
