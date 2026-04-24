# Rule Coverage Status

## Implemented

Current implementation is sufficient for parser validation, replay consistency checks, and first-pass supervised sample building.

### Parsing and Replay

- Tenhou XML parsing for `GO`, `UN`, `TAIKYOKU`, `INIT`, draws, discards, `N`, `DORA`, `REACH`, `AGARI`, `RYUUKYOKU`
- Tenhou meld decoding for `chi`, `pon`, `daiminkan`, `ankan`, `kakan`
- Four-player replay state with:
  - exact hand tiles
  - red-five bookkeeping
  - discards
  - melds
  - riichi declaration state
  - basic score tracking for riichi stick deduction
- Replay validation for:
  - initial 13-tile hands
  - discard must exist in hand
  - meld consumed tiles must exist in hand
  - meld source and called tile consistency
  - `kakan` upgrading an existing `pon`

### Sample Building

- `turn` samples:
  - `discard`
  - `riichi_discard`
- `response` samples:
  - `pass`
  - `chi`
  - `pon`
  - `daiminkan`
  - `ron`
- Exact red-tile action handling in legal candidate generation

### Hand Checks

- Standard agari detection
- Chiitoitsu detection
- Kokushi musou detection
- Basic tenpai-after-discard check for `riichi_discard` legality

## Not Yet Implemented

These items are intentionally not modeled yet and should be treated as open work, not implied support.

### Turn Actions

- `tsumo`
- `ankan` as a turn decision candidate
- `kakan` as a turn decision candidate
- `kyuushu_kyuuhai`
- exhaustive turn-action arbitration across all legal self-turn actions

### Response Actions

- `chankan`
- double or multi-ron edge handling
- full furiten-aware `ron` filtering

### Riichi and Win Restrictions

- full riichi legality beyond current approximation
  - ippatsu state
  - dead-wall / remaining-tile edge checks
  - exhaustive lock after riichi
- complete furiten handling
- nagashi mangan
- exhaustive abortive draw handling

### Scoring and Round Resolution

- full yaku evaluation
- fu / han scoring
- point movement validation
- ura dora, kan dora, kan ura dora effects
- placement / end-of-game settlement checks

### Dataset and Modeling Gaps

- response samples for all unsupported edge cases above
- serialized event shards (`bronze_events`) are not written yet
- parquet writing currently starts from raw XML parsing plus in-memory replay, not from persisted `bronze_events`

## Practical Meaning

At this stage, the repository can reliably:

- parse real Tenhou logs
- replay and validate core action legality
- build first-pass `turn` and `response` supervised samples

It should not yet be described as a full Riichi rules engine or a complete Mahjong Soul action simulator.
