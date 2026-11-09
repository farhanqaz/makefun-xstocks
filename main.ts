// make.fun xstocks-movers-forecast-01 entry.
// Reads task.json and bars/<SYMBOL>.ndjson, runs a seeded Monte Carlo per window day with a one-factor
// correlation model, and writes a forecast with place probabilities, a forecast order and return-bin sizes.
// Standard library only and deterministic: the same input always produces the same bytes.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const HOUR_MS = 3600000;
const RETURN_LOOKBACK = 480;
const DEFAULT_DAILY_VOL = 0.02;
const MIN_DAILY_VOL = 0.005;
const HOURS_PER_DAY = 24;
const CORRELATION = 0.45;
const DRAWS = 60000;
const PLACE_CAP = 0.98;
const PLACE_MIN_PROBABILITY = 0.001;
const SIZE_SMOOTHING = 0.5;
const SIZE_MIN_PROBABILITY = 0.0005;

interface Bar {
  t: number;
  c: number;
}

interface WindowDay {
  date: string;
  previous: string;
  close: string;
  twap_from?: string;
  twap_until?: string;
}

interface Task {
  places: number;
  bin_edges: number[];
  output?: string;
  history_until: string;
  twap: { time_zone: string };
  window: { days: WindowDay[] };
  universe: { symbol: string; file: string }[];
}

interface Name {
  symbol: string;
  dailyVol: number;
  lastClose: number;
  lastBarTime: number;
  previousClose: number | null;
}

interface DayForecast {
  date: string;
  probabilities: { symbol: string; place: number; p: number }[];
  order: string[];
  sizes: { symbol: string; bin: number; p: number }[];
}

// Mulberry32 PRNG: returns uniforms in [0, 1), fully determined by the seed.
function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
    mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

// Box-Muller transform: turns pairs of uniforms into standard normals, caching the spare value.
function standardNormals(uniform: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    let u1 = uniform();
    while (u1 <= 0) u1 = uniform();
    const u2 = uniform();
    const radius = Math.sqrt(-2 * Math.log(u1));
    spare = radius * Math.sin(2 * Math.PI * u2);
    return radius * Math.cos(2 * Math.PI * u2);
  };
}

// Offset of a time zone from UTC at the given instant, in milliseconds.
function zoneOffsetMs(instant: number, timeZone: string): number {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hourCycle: 'h23',
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
    second: 'numeric',
  }).formatToParts(new Date(instant));
  const field = (type: string) => Number(parts.find((part) => part.type === type)!.value);
  const wallClockAsUtc = Date.UTC(
    field('year'),
    field('month') - 1,
    field('day'),
    field('hour'),
    field('minute'),
    field('second'),
  );
  return wallClockAsUtc - instant;
}

// Converts a wall-clock time (YYYY-MM-DD and HH:MM in the given zone) to a UTC instant in ms.
function zonedClockToUtc(date: string, clock: string, timeZone: string): number {
  const [year, month, day] = date.split('-').map(Number);
  const [hour, minute] = clock.split(':').map(Number);
  const wallClock = Date.UTC(year, month - 1, day, hour, minute);
  const firstGuess = wallClock - zoneOffsetMs(wallClock, timeZone);
  return wallClock - zoneOffsetMs(firstGuess, timeZone);
}

// Loads a bars file and sorts it oldest first by hour-close time.
function loadBars(file: string): Bar[] {
  const lines = readFileSync(file, 'utf8').split('\n').filter((line) => line.trim() !== '');
  const bars = lines.map((line) => {
    const row = JSON.parse(line);
    return { t: Number(row.t), c: Number(row.c) };
  });
  return bars.sort((left, right) => left.t - right.t);
}

// House baseline daily vol: sqrt(24 * mean(r^2)) over the last 480 log returns between bars one hour apart.
function dailyVol(bars: Bar[]): number {
  const returns: number[] = [];
  for (let index = 1; index < bars.length; index++) {
    if (bars[index].t - bars[index - 1].t === HOUR_MS) {
      returns.push(Math.log(bars[index].c / bars[index - 1].c));
    }
  }
  if (returns.length < 2) return DEFAULT_DAILY_VOL;
  const recent = returns.slice(-RETURN_LOOKBACK);
  const meanSquare = recent.reduce((total, value) => total + value * value, 0) / recent.length;
  return Math.max(MIN_DAILY_VOL, Math.sqrt(HOURS_PER_DAY * meanSquare));
}

// Previous close bar: the bar at the previous session's close instant, or the last bar before it.
function previousCloseBar(bars: Bar[], day: WindowDay, timeZone: string): Bar | null {
  const closeInstant = zonedClockToUtc(day.previous, day.close, timeZone);
  let found: Bar | null = null;
  for (const bar of bars) {
    if (bar.t <= closeInstant) found = bar;
  }
  return found;
}

// Day close instant: the window entry's twap_until, else twap_from, else the history end.
function dayCloseInstant(day: WindowDay, task: Task): number {
  return Date.parse(day.twap_until ?? day.twap_from ?? task.history_until);
}

// Bin index for a return: 0 below edges[0], i for edges[i-1] <= r < edges[i], edges.length at or above the last edge.
function binOf(value: number, edges: number[]): number {
  let bin = 0;
  for (const edge of edges) {
    if (value >= edge) bin++;
  }
  return bin;
}

function compareSymbols(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function forecastDay(day: WindowDay, dayIndex: number, names: Name[], task: Task): DayForecast {
  const binCount = task.bin_edges.length + 1;

  // Day 1 uses its known part (log return since the previous close, or 0 if that bar is missing) as the mean
  // and sqrt(hours left / 24) of the daily sd, with hours left measured from the last bar to the day's close.
  // Later days use mean 0 and the full daily sd.
  const means: number[] = [];
  const sds: number[] = [];
  for (const name of names) {
    if (dayIndex === 0) {
      const hoursLeft = Math.min(HOURS_PER_DAY, Math.max(1, (dayCloseInstant(day, task) - name.lastBarTime) / HOUR_MS));
      means.push(name.previousClose === null ? 0 : Math.log(name.lastClose / name.previousClose));
      sds.push(name.dailyVol * Math.sqrt(hoursLeft / HOURS_PER_DAY));
    } else {
      means.push(0);
      sds.push(name.dailyVol);
    }
  }

  // Each day gets its own seed, taken from its date digits (e.g. 2026-11-03 -> 20261103).
  const normal = standardNormals(mulberry32(Number(day.date.replace(/\D/g, ''))));

  const placeCounts: number[][] = [];
  for (let place = 0; place < task.places; place++) placeCounts.push(new Array(names.length).fill(0));
  const sizeCounts: number[][] = names.map(() => new Array(binCount).fill(0));

  const returns: number[] = new Array(names.length).fill(0);
  for (let draw = 0; draw < DRAWS; draw++) {
    // One-factor model: u_i = sqrt(RHO) * z + sqrt(1 - RHO) * e_i, return_i = mean_i + sd_i * u_i.
    const common = normal();
    for (let index = 0; index < names.length; index++) {
      const specific = normal();
      const standardised = Math.sqrt(CORRELATION) * common + Math.sqrt(1 - CORRELATION) * specific;
      returns[index] = means[index] + sds[index] * standardised;
    }

    // Place 1 is the largest return. Names are already in symbol order, so index order breaks ties.
    const ranking = names.map((_, index) => index).sort((left, right) => returns[right] - returns[left] || left - right);
    for (let place = 0; place < task.places; place++) placeCounts[place][ranking[place]]++;

    for (let index = 0; index < names.length; index++) {
      sizeCounts[index][binOf(returns[index], task.bin_edges)]++;
    }
  }

  // Probabilities: scale a place's row down to PLACE_CAP when it exceeds it, then drop tiny rows.
  const probabilities: DayForecast['probabilities'] = [];
  for (let place = 0; place < task.places; place++) {
    const rowTotal = placeCounts[place].reduce((total, count) => total + count / DRAWS, 0);
    const scale = rowTotal > PLACE_CAP ? PLACE_CAP / rowTotal : 1;
    names.forEach((name, index) => {
      const probability = (placeCounts[place][index] / DRAWS) * scale;
      if (probability >= PLACE_MIN_PROBABILITY) {
        probabilities.push({ symbol: name.symbol, place: place + 1, p: probability });
      }
    });
  }
  probabilities.sort((left, right) => left.place - right.place || compareSymbols(left.symbol, right.symbol));

  // Order: place by place, take the unplaced name with the largest count at that place (ties by symbol).
  const order: string[] = [];
  const placed = new Set<number>();
  for (let place = 0; place < task.places; place++) {
    let best = -1;
    for (let index = 0; index < names.length; index++) {
      if (placed.has(index)) continue;
      if (best < 0 || placeCounts[place][index] > placeCounts[place][best]) best = index;
    }
    placed.add(best);
    order.push(names[best].symbol);
  }

  // Sizes: add SIZE_SMOOTHING to each bin count, normalise per name, then drop tiny rows.
  const smoothedTotal = DRAWS + SIZE_SMOOTHING * binCount;
  const sizes: DayForecast['sizes'] = [];
  for (let bin = 0; bin < binCount; bin++) {
    names.forEach((name, index) => {
      const probability = (sizeCounts[index][bin] + SIZE_SMOOTHING) / smoothedTotal;
      if (probability >= SIZE_MIN_PROBABILITY) sizes.push({ symbol: name.symbol, bin, p: probability });
    });
  }
  sizes.sort((left, right) => left.bin - right.bin || compareSymbols(left.symbol, right.symbol));

  return { date: day.date, probabilities, order, sizes };
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  const task: Task = JSON.parse(readFileSync(path.join(inputDir, 'task.json'), 'utf8'));

  // Names sorted by symbol, so index order is the symbol tie-break everywhere.
  const universe = [...task.universe].sort((left, right) => compareSymbols(left.symbol, right.symbol));
  const firstDay = task.window.days[0];
  const names: Name[] = universe.map((entry) => {
    const bars = loadBars(path.join(inputDir, entry.file));
    const previousBar = previousCloseBar(bars, firstDay, task.twap.time_zone);
    return {
      symbol: entry.symbol,
      dailyVol: dailyVol(bars),
      lastClose: bars[bars.length - 1].c,
      lastBarTime: bars[bars.length - 1].t,
      previousClose: previousBar === null ? null : previousBar.c,
    };
  });

  const days = task.window.days.map((day, dayIndex) => forecastDay(day, dayIndex, names, task));
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(path.join(outputDir, task.output ?? 'forecast.json'), JSON.stringify({ days }, null, 2) + '\n', 'utf8');
}

main();
