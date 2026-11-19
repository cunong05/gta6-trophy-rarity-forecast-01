// gta6-trophy-rarity-forecast-01 — logit-growth forecaster (Node 22, stdlib only).
//
// Model, in one paragraph: work in logit space on each trophy's earn rate.
// From the case's distinct reads (consecutive identical reads collapsed) fit,
// per trophy, a weighted least-squares line of logit(rate) against
// log(1 + hours since release), with the slope shrunk toward the tier median
// and the extrapolation damped for long horizons, capped at a tier ceiling.
// A players-tracked power-law fit gives N(target); the forecast rate is
// floored so implied earners never fall below the last observed earners.
// Exophase refreshes a game's counts in jumps, hours apart: when the page
// has been flat/stale, a refresh is likely before the target, and a refresh
// dilutes rates (launch night: 638 -> 3,868 players, new cohort earning at
// ~7% of the old cohort's rate, so non-100% rates fell to ~22% of their
// value). A stale-refresh shift, sized by how overdue a refresh is, moves
// the median down by that dilution, and widens the spread. Spread grows
// with sqrt(horizon) and with staleness, so a 90-minute practice spread is
// narrow-ish while the 6,510-minute final spread is wide.
//
// Deterministic: no clock, no randomness in the output path; only the case
// file's `at` / `target` / `horizon_minutes` are used. One .out per .in,
// trophies sorted by id, quantiles rounded to 4 decimals, non-decreasing.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels the contest scores, in order. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at LEVELS. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release-day midnight UTC; time axis origin for the growth fits. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

/** Week-one ceiling (percent) per tier: loose caps against runaway extrapolation. */
const TIER_CEILING: Record<string, number> = {
  platinum: 3,
  gold: 15,
  silver: 30,
  bronze: 80,
};
const ceilingOf = (tier: string): number => TIER_CEILING[tier] ?? 80;
/** Prior players multiplier of one Exophase refresh (launch night: 3868/638). */
const PRIOR_REFRESH_MULT = 6.06;
/** Prior fraction of the old cohort's rate at which a fresh cohort earns. */
const PRIOR_NEW_COHORT_RATE = 0.067;
/** Prior typical gap between Exophase refreshes, hours (launch night ~4.7h). */
const PRIOR_REFRESH_GAP_H = 5;
/** Random-walk spread, logit units per sqrt(day). */
const TREND_SIGMA_PER_SQRT_DAY = 0.45;
/** Extra spread when a stale refresh is likely, logit units at full weight. */
const REFRESH_SIGMA = 0.9;

interface TrophyMeta {
  id: number;
  tier: string;
}
interface Read {
  ms: number;
  players: number;
  rates: Array<number | null>;
}
interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: TrophyMeta[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct);
  return Math.log(p / (100 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
/** Hours from release midnight to a timestamp. */
const hoursSinceRelease = (ms: number): number =>
  Math.max(0, (ms - RELEASE_MS) / 3_600_000);
/** Log-time axis used by every fit: log(1 + hours). */
const logTime = (ms: number): number => Math.log1p(hoursSinceRelease(ms));

/** Weighted least squares line y = a + b*x; returns [a, b]. */
function wls(xs: number[], ys: number[], ws: number[]): [number, number] {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < xs.length; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - mx;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  return [my - b * mx, b];
}

/** Collapse consecutive reads that change nothing (same players, same rates). */
function distinctReads(history: Read[]): Read[] {
  const out: Read[] = [];
  for (const r of history) {
    const prev = out[out.length - 1];
    if (
      prev !== undefined &&
      prev.players === r.players &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      continue;
    }
    out.push(r);
  }
  return out;
}

/** Two-parameter WLS for y = a*u + b*v; returns [a, b, sse]. */
function wls2(
  us: number[],
  vs: number[],
  ys: number[],
  ws: number[],
): [number, number, number] {
  let suu = 0;
  let suv = 0;
  let svv = 0;
  let su = 0;
  let sv = 0;
  let sy = 0;
  let sw = 0;
  for (let i = 0; i < ys.length; i++) {
    const w = ws[i];
    suu += w * us[i] * us[i];
    suv += w * us[i] * vs[i];
    svv += w * vs[i] * vs[i];
    su += w * us[i];
    sv += w * vs[i];
    sy += w * ys[i];
    sw += w;
  }
  // Solve the 2x2 normal equations for the basis coefficients.
  const det = suu * svv - suv * suv;
  let a = 0;
  let b = 0;
  if (Math.abs(det) > 1e-12) {
    const rhsU = ys.reduce((s, y, i) => s + ws[i] * us[i] * y, 0);
    const rhsV = ys.reduce((s, y, i) => s + ws[i] * vs[i] * y, 0);
    a = (rhsU * svv - rhsV * suv) / det;
    b = (rhsV * suu - rhsU * suv) / det;
  } else {
    b = sy / Math.max(sw, 1e-9);
  }
  let sse = 0;
  for (let i = 0; i < ys.length; i++) {
    const r = ys[i] - (a * us[i] + b * vs[i]);
    sse += ws[i] * r * r;
  }
  return [a, b, sse];
}

/**
 * Saturating growth fit in real time: L(h) = L0*e^(-k*h) + Linf*(1-e^(-k*h)).
 * Grid-searches k; returns the forecast logit at targetHours, or null when
 * the history is too thin. A second shape to hedge the log-time-linear fit:
 * the two disagree most exactly when the truth saturates, which is when a
 * straight log-time line overshoots or undershoots worst.
 */
function saturatingForecast(
  hours: number[],
  ys: number[],
  ws: number[],
  targetHours: number,
  ceilingLogit: number,
): number | null {
  if (hours.length < 3) return null;
  const ks = [1 / 6, 1 / 12, 1 / 24, 1 / 48, 1 / 96, 1 / 192, 1 / 384, 1 / 768];
  let best: { sse: number; l0: number; linf: number; k: number } | null = null;
  for (const k of ks) {
    const us = hours.map((h) => Math.exp(-k * h));
    const vs = us.map((u) => 1 - u);
    const [l0, linf, sse] = wls2(us, vs, ys, ws);
    if (best === null || sse < best.sse) best = { sse, l0, linf, k };
  }
  if (best === null) return null;
  const linf = Math.min(best.linf, ceilingLogit);
  const uT = Math.exp(-best.k * targetHours);
  return best.l0 * uT + linf * (1 - uT);
}

function lastKnownRate(history: Read[], idx: number): number | null {
  for (let i = history.length - 1; i >= 0; i--) {
    const v = history[i].rates[idx];
    if (v !== null && v !== undefined && Number.isFinite(v)) return v;
  }
  return null;
}

/**
 * Estimate refresh behaviour from the distinct reads: for each transition
 * where players_tracked jumps >= 1.5x, the players multiplier M and the
 * new cohort's earning fraction k (from the change in total earners vs
 * what the new players would have earned at the old rates). Falls back to
 * the launch-night priors when no refresh is visible in the history.
 */
function estimateRefresh(
  distinct: Read[],
): { mult: number; newRateFrac: number; gapHours: number } {
  const mults: number[] = [];
  const fracs: number[] = [];
  const gaps: number[] = [];
  for (let i = 1; i < distinct.length; i++) {
    const prev = distinct[i - 1];
    const cur = distinct[i];
    gaps.push((cur.ms - prev.ms) / 3_600_000);
    if (cur.players >= 1.5 * prev.players && prev.players > 0) {
      mults.push(cur.players / prev.players);
      let oldEarners = 0;
      let expectedNew = 0;
      let newEarners = 0;
      for (let j = 0; j < cur.rates.length; j++) {
        const ro = prev.rates[j];
        const rn = cur.rates[j];
        if (ro === null || rn === null) continue;
        oldEarners += (ro * prev.players) / 100;
        newEarners += (rn * cur.players) / 100;
        expectedNew += (ro * (cur.players - prev.players)) / 100;
      }
      if (expectedNew > 0) {
        fracs.push(
          Math.min(1, Math.max(0, (newEarners - oldEarners) / expectedNew)),
        );
      }
    }
  }
  return {
    mult: mults.length > 0 ? median(mults) : PRIOR_REFRESH_MULT,
    newRateFrac:
      fracs.length > 0 ? median(fracs) : PRIOR_NEW_COHORT_RATE,
    gapHours: gaps.length > 0 ? median(gaps) : PRIOR_REFRESH_GAP_H,
  };
}

/** Forecast players_tracked at the target by a power law in log-time. */
function forecastPlayers(
  distinct: Read[],
  targetMs: number,
): { nTarget: number; nLast: number } {
  const nLast = distinct[distinct.length - 1].players;
  if (distinct.length < 2) return { nTarget: nLast, nLast };
  const xs: number[] = [];
  const ys: number[] = [];
  const ws: number[] = [];
  distinct.forEach((r, i) => {
    if (r.players > 0) {
      xs.push(logTime(r.ms));
      ys.push(Math.log(r.players));
      ws.push(Math.pow(0.9, distinct.length - 1 - i));
    }
  });
  if (xs.length < 2) return { nTarget: nLast, nLast };
  const [a, bRaw] = wls(xs, ys, ws);
  const b = Math.min(3, Math.max(0, bRaw));
  const logN = a + b * logTime(targetMs);
  // Power laws can run away over 4.5 days; cap total growth at 50x.
  const nTarget = Math.min(
    nLast * 50,
    Math.max(nLast, Math.exp(logN)),
  );
  return { nTarget, nLast };
}

/**
 * Forecast one trophy. Returns the median logit and the logit spread.
 * `tierSlopes` is the median fitted slope per tier (shrinkage prior).
 */
function forecastTrophy(
  idx: number,
  trophy: TrophyMeta,
  c: CaseFile,
  distinct: Read[],
  tierSlopes: Map<string, number>,
  refresh: { mult: number; newRateFrac: number; gapHours: number },
  playersFc: { nTarget: number; nLast: number },
  staleHours: number,
  flatFactor: number,
): { medLogit: number; sigma: number } {
  const last = lastKnownRate(c.history, idx);
  const lastRate = last === null ? 0.5 : last;
  const lastLogit = logit(lastRate);
  const targetMs = Date.parse(c.target);
  const horizonDays = Math.max(0, c.horizon_minutes / 1440);

  // Per-trophy trend: logit(rate) vs log-time over distinct reads.
  const xs: number[] = [];
  const ys: number[] = [];
  const ws: number[] = [];
  distinct.forEach((r, i) => {
    const v = r.rates[idx];
    if (v !== null && v !== undefined && Number.isFinite(v)) {
      xs.push(logTime(r.ms));
      ys.push(logit(v));
      ws.push(Math.pow(0.9, distinct.length - 1 - i));
    }
  });
  let medLogit = lastLogit;
  let fitUncertainty = 0.1;
  if (xs.length >= 2) {
    const [a, bRaw] = wls(xs, ys, ws);
    const prior = tierSlopes.get(trophy.tier) ?? 0;
    // Shrink the slope toward the tier median (3 pseudo-observations).
    const nEff = xs.length;
    const b = (nEff * bRaw + 3 * prior) / (nEff + 3);
    const xFirst = xs[0];
    const xLast = xs[xs.length - 1];
    const xTarget = logTime(targetMs);
    // Damped extrapolation: trust the fitted move in proportion to how
    // much log-time range was actually observed vs how far past it the
    // target sits (observed range / (observed range + gap beyond it)).
    const xRange = Math.max(0, xLast - xFirst);
    const gap = Math.max(0, xTarget - xLast);
    const damp = xRange > 1e-9 ? xRange / (xRange + gap) : 0;
    let logTimeFc = a + b * xLast + damp * b * gap;
    // Saturating-growth fit in real time, blended 50/50: the hedge
    // against the log-time line when the truth bends toward a ceiling.
    const satHours: number[] = [];
    const satYs: number[] = [];
    const satWs: number[] = [];
    distinct.forEach((r, i) => {
      const v = r.rates[idx];
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        satHours.push(hoursSinceRelease(r.ms));
        satYs.push(logit(v));
        satWs.push(Math.pow(0.9, distinct.length - 1 - i));
      }
    });
    const sat = saturatingForecast(
      satHours,
      satYs,
      satWs,
      hoursSinceRelease(targetMs),
      logit(ceilingOf(trophy.tier)),
    );
    if (sat !== null && Number.isFinite(sat)) {
      logTimeFc = 0.5 * logTimeFc + 0.5 * sat;
    }
    medLogit = logTimeFc;
    // Residual scatter around the fit, as extra uncertainty.
    let rss = 0;
    let wsum = 0;
    for (let i = 0; i < xs.length; i++) {
      const r = ys[i] - (a + b * xs[i]);
      rss += ws[i] * r * r;
      wsum += ws[i];
    }
    fitUncertainty = 0.1 + Math.sqrt(rss / Math.max(wsum, 1e-9));
  }

  // Tier ceiling (a trophy cannot plausibly exceed its tier's week-one cap).
  const ceiling = ceilingOf(trophy.tier);
  medLogit = Math.min(medLogit, logit(ceiling));
  // Earners floor: implied earners at the target cannot fall below the
  // earners already observed (players only join; earners only grow).
  if (playersFc.nTarget > playersFc.nLast && lastRate < 100) {
    const earnersLast = (lastRate * playersFc.nLast) / 100;
    const floorRate = (earnersLast / playersFc.nTarget) * 100;
    medLogit = Math.max(medLogit, logit(floorRate));
  }

  // Stale-refresh shift: ONLY when the history is (near-)flat does the
  // trend fit carry no signal about the next refresh, so a dilution shift
  // is warranted. When reads move, refreshes are already priced into the
  // fitted trend; applying another dilution would double-count them.
  // Weight grows with (staleness + horizon) relative to the typical gap.
  let refreshWeight = 0;
  let refreshSigma = 0;
  if (lastRate < 100 && flatFactor > 0) {
    const horizonHours = c.horizon_minutes / 60;
    const gap = Math.max(1, refresh.gapHours);
    refreshWeight =
      flatFactor * (1 - Math.exp(-(staleHours + horizonHours) / gap));
    const dilution =
      (1 + (refresh.mult - 1) * refresh.newRateFrac) / refresh.mult;
    medLogit += refreshWeight * Math.log(dilution);
    refreshSigma = REFRESH_SIGMA * refreshWeight;
  }

  // Spread: random-walk growth with horizon + fit uncertainty +
  // refresh uncertainty + a small rarity term for very rare trophies.
  const trendSigma =
    0.15 +
    TREND_SIGMA_PER_SQRT_DAY * Math.sqrt(horizonDays) +
    fitUncertainty * 0.5;
  const raritySigma = lastRate < 1 ? 0.1 : lastRate < 5 ? 0.05 : 0;
  const sigma = Math.sqrt(
    trendSigma * trendSigma +
      refreshSigma * refreshSigma +
      raritySigma * raritySigma,
  );
  return { medLogit, sigma };
}

/** Forecast one whole case into the output row list. */
function forecastCase(c: CaseFile): Array<{ id: number; q: number[] }> {
  const history: Read[] = c.history.map((r) => ({
    ms: Date.parse(r.at),
    players: r.players_tracked,
    rates: r.rates,
  }));
  const distinct = distinctReads(history);
  const atMs = Date.parse(c.at);

  // Staleness: hours since the last read that actually changed anything.
  const lastChangeMs =
    distinct.length > 0 ? distinct[distinct.length - 1].ms : atMs;
  const staleHours = Math.max(0, (atMs - lastChangeMs) / 3_600_000);

  const refresh = estimateRefresh(distinct);
  const playersFc = forecastPlayers(
    distinct,
    Date.parse(c.target),
  );

  // Flatness: median absolute logit move from the first to the last
  // distinct read, across trophies. Near zero means the page never moved
  // and the trend fit is silent (practice regime); large means refreshes
  // and growth are already visible in the history (final regime).
  const moves: number[] = [];
  if (distinct.length >= 2) {
    const first = distinct[0];
    const lastD = distinct[distinct.length - 1];
    c.trophies.forEach((_, idx) => {
      const a = first.rates[idx];
      const b = lastD.rates[idx];
      if (
        a !== null && b !== null &&
        a !== undefined && b !== undefined &&
        Number.isFinite(a) && Number.isFinite(b)
      ) {
        moves.push(Math.abs(logit(b) - logit(a)));
      }
    });
  }
  const moveMed = median(moves);
  const flatFactor =
    moveMed < 0.05 ? 1 : moveMed > 0.5 ? 0 : (0.5 - moveMed) / 0.45;

  // Tier-median fitted slopes, for shrinkage (needs a first pass).
  const rawSlopes = new Map<string, number[]>();
  c.trophies.forEach((t, idx) => {
    const xs: number[] = [];
    const ys: number[] = [];
    const ws: number[] = [];
    distinct.forEach((r, i) => {
      const v = r.rates[idx];
      if (v !== null && v !== undefined && Number.isFinite(v)) {
        xs.push(logTime(r.ms));
        ys.push(logit(v));
        ws.push(Math.pow(0.9, distinct.length - 1 - i));
      }
    });
    if (xs.length >= 2) {
      const [, b] = wls(xs, ys, ws);
      const arr = rawSlopes.get(t.tier) ?? [];
      arr.push(b);
      rawSlopes.set(t.tier, arr);
    }
  });
  const tierSlopes = new Map<string, number>();
  for (const [tier, arr] of rawSlopes) tierSlopes.set(tier, median(arr));

  const rows = c.trophies
    .map((t, idx) => ({ t, idx }))
    .sort((a, b) => a.t.id - b.t.id)
    .map(({ t, idx }) => {
      const { medLogit, sigma } = forecastTrophy(
        idx,
        t,
        c,
        distinct,
        tierSlopes,
        refresh,
        playersFc,
        staleHours,
        flatFactor,
      );
      // Quantiles from the median logit and spread; enforce ordering
      // after rounding (rounding can tie adjacent quantiles, never invert).
      const q: number[] = [];
      let prev = -Infinity;
      for (const z of Z) {
        const v = round4(clampPct(invLogit(medLogit + z * sigma)));
        q.push(Math.max(v, prev));
        prev = q[q.length - 1];
      }
      return { id: t.id, q };
    });
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((t, j) => t !== LEVELS[j])
  ) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (c.history.length === 0) {
    throw new Error(`${file} has no history`);
  }
  const trophies = forecastCase(c);
  writeFileSync(
    join(OUTPUT_DIR, `${id}.out`),
    `${JSON.stringify({ trophies })}\n`,
  );
}
