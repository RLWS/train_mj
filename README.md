# Mahjong AI Training

This repository contains the training pipeline for a four-player Riichi Mahjong AI
targeting Mahjong Soul, using Tenhou logs as the primary supervised-learning data source.

The current focus is building the engineering foundation:

- Tenhou XML parsing and validated event replay.
- Legal action generation for turn and response decisions.
- Shared encoder with task heads for `turn`, `response`, and `value`.
- Local smoke tests on RTX 4070 before full training on the P800 machine.

See `MAHJONG_AI_PLAN.md` and `MAHJONG_AI_ENGINEERING.md` for the detailed plan.
