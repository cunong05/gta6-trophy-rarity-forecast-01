# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cunong05/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/cunong05/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks who will have earned it at
2026-11-26T00:00:00Z — a week after launch — as five quantiles per trophy,
scored by pinball loss on the logit scale.
Entry for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Model

`main.ts` (Node 22, stdlib only) works in logit space, one trophy at a time:

- **Trend fit** — weighted least squares of logit(rate) against
  log(1 + hours since release) over the case's distinct reads, slope shrunk
  toward the tier median, extrapolation damped by how far the target sits
  past the observed log-time range, capped at a tier ceiling (platinum 3%,
  gold 15%, silver 30%, bronze 80%).
- **Saturating-growth fit** — `L(h) = L0·e^(−kh) + Linf·(1−e^(−kh))` in real
  time, grid-searched over k, blended 50/50 with the trend fit as a hedge
  against the log-time line when the truth bends toward a ceiling.
- **Players / earners floor** — a power-law fit of players-tracked forecasts
  N(target); the forecast rate never implies fewer earners than already
  observed.
- **Stale-refresh correction** — Exophase refreshes a game's counts in jumps
  hours apart (launch night: 638 → 3,868 players, new cohort earning at
  ~7% of the old rate, so rates fell to ~22%). When the page has been
  flat/stale, the median is shifted down by that dilution and the spread
  widened; when reads already move, refreshes are priced into the trend
  and no extra dilution is applied.
- **Spread** — grows with √horizon plus fit, refresh and rarity terms, so
  the 90-minute practice spread is narrow while the 6,510-minute final
  spread is wide. Deterministic: no clock or randomness in the output path.

## Validation (local, practice cases p00/p01)

Proxy answers reconstructed from the kit's own figures (04:30Z read:
3,868 players, 4,276 trophies earned; reproduces the stated typical miss
0.68 points and persistence loss ≈1.4–1.5):

| forecaster | p00 | p01 |
|---|---|---|
| persistence (house baseline) | 1.401 | 1.401 |
| logit-linear trend | 1.401 | 1.401 |
| **this model** | **0.450** | **0.415** |

Synthetic long-horizon check (2 days of moving history, target +4.5 days):
model 0.286 vs persistence 0.414. Spread (log q95/q05) widens from 2.06
at 90 min to 3.12 at 6,510 min.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input dir, writes one `<id>.out` per case:
`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`, 55 trophies,
quantiles non-decreasing, 0–100.

---

<sub>Entered on [make.fun](https://make.fun/?ref=0ene8b3q&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
