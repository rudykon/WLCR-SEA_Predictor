/* WLCR-SEA browser inference. No network, DOM, or server inference in this module.
 * Model semantics follow experiments/wlcr_sea_model.py. Public float32 weights;
 * JavaScript accumulations may differ slightly from a PyTorch backend.
 */
export const HOURS = 336;
export const HORIZON = 24;
export const METRICS = 4;
export const EXPERTS = 8;
export const REVISION = 'ensemble-v1';
export const HEADER = ['时间', '小区名称', '小区上行平均激活用户数', '小区下行平均激活用户数', '下行平均使用的PRB个数', '上行平均使用的PRB个数'];
const f32 = Math.fround;
const fail = (code) => { throw new Error(code); };
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const product = (shape) => shape.reduce((a, b) => a * b, 1);
export async function sha256(bytes) {
  return Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
}

// RFC-style quoted CSV, including doubled quotes, BOMs, CRLF and quoted cells.
export function parseCSV(text) {
  text = text.replace(/^\uFEFF/, '');
  const rows = []; let row = [], cell = '', quoted = false, closed = false, started = false;
  const field = () => { row.push(cell); cell = ''; closed = false; started = false; };
  const finish = () => { field(); if (row.some(x => x.trim())) rows.push(row); row = []; };
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { cell += '"'; i++; }
        else { quoted = false; closed = true; }
      } else cell += c;
    } else if (closed && c !== ',' && c !== '\r' && c !== '\n') fail('invalid_csv');
    else if (c === '"') { if (started) fail('invalid_csv'); quoted = true; started = true; }
    else if (c === ',') field();
    else if (c === '\r' || c === '\n') { if (c === '\r' && text[i + 1] === '\n') i++; finish(); }
    else { cell += c; started = true; }
  }
  if (quoted) fail('invalid_csv');
  if (cell || row.length || started || closed) finish();
  return rows;
}

export function parseRequest(text) {
  const rows = parseCSV(text);
  if (!rows.length) fail('empty_csv');
  if (rows[0].length !== 6 || rows[0].some((x, i) => x !== HEADER[i])) fail('invalid_header');
  if (rows.length !== HOURS + 1) fail('wrong_row_count');
  const values = [], times = [], counts = [0, 0, 0, 0]; let cellName;
  const numeric = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;
  for (const row of rows.slice(1)) {
    if (row.length !== 6) fail('invalid_columns');
    const cell = row[1].trim();
    if (!cell) fail('empty_cell');
    if (cellName !== undefined && cellName !== cell) fail('mixed_cells');
    cellName = cell;
    const match = row[0].trim().match(/^(\d{4})\/(\d{2})\/(\d{2}) (\d{2}):(\d{2})$/);
    if (!match) fail('invalid_timestamp');
    const [year, month, day, hour, minute] = match.slice(1).map(Number);
    const date = new Date(0); date.setUTCFullYear(year, month - 1, day); date.setUTCHours(hour, minute, 0, 0);
    if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month - 1 || date.getUTCDate() !== day || date.getUTCHours() !== hour || date.getUTCMinutes() !== minute) fail('invalid_timestamp');
    const time = date.getTime();
    if (times.length && time - times.at(-1) !== 3600000) fail('non_contiguous_timestamps');
    times.push(time);
    values.push(row.slice(2).map((raw, q) => {
      const s = raw.trim();
      if (!s || s.toUpperCase() === 'NIL') return null;
      if (!numeric.test(s)) fail('invalid_metric');
      const value = f32(Number(s));
      if (!Number.isFinite(value) || value < 0) fail('invalid_metric');
      counts[q]++; return value;
    }));
  }
  if (counts.some(x => x === 0)) fail('indicator_unobserved');
  return { cell: cellName, times, values };
}

export async function stableUniform(...tokens) {
  const bytes = new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(tokens.join('|'))));
  return Number(new DataView(bytes.buffer).getBigUint64(0, false)) / (2 ** 64);
}
const roundEven = (x) => { const base = Math.floor(x); return x - base === 0.5 ? base + (base % 2) : Math.round(x); };
export async function corruptRequest(request, scenario = 'none', rate = 0) {
  if (!['none', 'mcar', 'block', 'recent_tail', 'asynchronous'].includes(scenario)) fail('unknown_scenario');
  if (!Number.isFinite(rate) || rate < 0 || rate > 0.8) fail('invalid_missing_rate');
  const removed = Array.from({ length: HOURS }, () => [false, false, false, false]);
  const actualRate = scenario === 'none' ? 0 : rate;
  if (actualRate > 0) {
    if (scenario === 'block' || scenario === 'recent_tail') {
      const length = Math.min(HOURS, Math.max(1, roundEven(actualRate * HOURS)));
      const start = scenario === 'recent_tail' ? HOURS - length : Math.min(HOURS - length, Math.floor(await stableUniform(2026, request.cell, 'block') * (HOURS - length + 1)));
      for (let i = start; i < start + length; i++) removed[i].fill(true);
    } else {
      await Promise.all(request.times.map(async (time, i) => {
        const absoluteHour = Math.floor(time / 3600000);
        if (scenario === 'mcar') removed[i].fill(await stableUniform(2026, request.cell, absoluteHour, 'mcar') < actualRate);
        else await Promise.all([0, 1, 2, 3].map(async q => { removed[i][q] = await stableUniform(2026, request.cell, absoluteHour, q, 'async') < actualRate; }));
      }));
    }
  }
  const counts = [0, 0, 0, 0]; let original = 0, newlyRemoved = 0;
  const effective = request.values.map((row, i) => row.map((x, q) => {
    if (x !== null) { original++; if (removed[i][q]) newlyRemoved++; }
    const result = removed[i][q] ? null : x;
    if (result !== null) counts[q]++;
    return result;
  }));
  if (counts.some(x => x === 0)) fail('indicator_removed');
  return { ...request, values: effective, originalValues: request.values, removed, scenario, requestedRate: actualRate, originalObserved: original, observed: counts.reduce((a, b) => a + b, 0), newlyRemoved, removedFraction: newlyRemoved / original };
}

export async function decodeModel(manifest, buffer) {
  if (manifest.schema !== 'wlcr-sea-browser-model/v1' || manifest.repo_id !== 'wlcr-sea-ensemble' || manifest.revision !== REVISION || manifest.variant !== 'A6_mixed_aug') fail('invalid_model');
  if (manifest.weights?.dtype !== 'float32-le' || buffer.byteLength !== manifest.weights.byte_length || buffer.byteLength % 4) fail('invalid_model');
  if (await sha256(buffer) !== manifest.weights.sha256) fail('model_checksum');
  if (manifest.members.length !== 5 || manifest.members.some((m, i) => m.seed !== 42 + i)) fail('invalid_model');
  const all = new Float32Array(buffer.byteLength / 4), view = new DataView(buffer);
  for (let i = 0; i < all.length; i++) { all[i] = view.getFloat32(i * 4, true); if (!Number.isFinite(all[i])) fail('invalid_model'); }
  const members = manifest.members.map(m => {
    const d = m.selected_config.token_dim, hidden = m.selected_config.hidden_dim, r = d / 4;
    if (![16, 32].includes(d) || ![32, 64].includes(hidden) || ![0.25, 0.5].includes(m.selected_config.residual_bound)) fail('invalid_model');
    const shapes = { prior_log: [24, 4], 'horizon_embedding.weight': [24, r], 'metric_embedding.weight': [4, r], 'expert_embedding.weight': [8, r], reliability_beta_raw: [], expert_distance: [8] };
    for (const [name, input, output] of [['query_network', 2 * r + 5, d], ['key_network', r + 4, d], ['residual_network', 2 * r + 7, 1]]) {
      shapes[`${name}.0.weight`] = [hidden, input]; shapes[`${name}.0.bias`] = [hidden];
      shapes[`${name}.2.weight`] = [output, hidden]; shapes[`${name}.2.bias`] = [output];
    }
    const tensors = {};
    for (const [key, shape] of Object.entries(shapes)) {
      const t = m.tensors[key];
      if (!t || JSON.stringify(t.shape) !== JSON.stringify(shape) || t.length !== product(shape) || !Number.isInteger(t.offset) || t.offset < 0 || t.offset + t.length > all.length) fail('invalid_model');
      tensors[key] = { shape, data: all.subarray(t.offset, t.offset + t.length) };
    }
    return { ...m, tensors };
  });
  return { manifest, members };
}

const median = (items) => { if (!items.length) return null; const a = [...items].sort((a, b) => a - b), i = Math.floor(a.length / 2); return a.length % 2 ? a[i] : f32((a[i - 1] + a[i]) / 2); };
const mean = (items) => items.reduce((a, b) => a + b, 0) / items.length;
const rowOf = (tensor, i) => tensor.data.subarray(i * tensor.shape[1], (i + 1) * tensor.shape[1]);
// Abramowitz–Stegun 7.1.26: erf absolute error < 1.5e-7. GELU uses erf, not tanh.
function gelu(x) {
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return f32(0.5 * x * (1 + Math.sign(x) * erf));
}
function linear(x, w, bias) {
  const out = new Float32Array(w.shape[0]);
  for (let o = 0; o < out.length; o++) {
    let sum = bias.data[o]; const start = o * x.length;
    for (let i = 0; i < x.length; i++) sum += w.data[start + i] * x[i];
    out[o] = sum;
  }
  return out;
}
function network(x, name, tensors) {
  const hidden = linear(x, tensors[`${name}.0.weight`], tensors[`${name}.0.bias`]);
  for (let i = 0; i < hidden.length; i++) hidden[i] = gelu(hidden[i]);
  return linear(hidden, tensors[`${name}.2.weight`], tensors[`${name}.2.bias`]);
}
export function entmax15(scores) {
  if (!scores.length || scores.some(x => !Number.isFinite(x))) fail('numerical_range');
  const largest = Math.max(...scores) / 2;
  const scaled = scores.map(x => x / 2 - largest), sorted = [...scaled].sort((a, b) => b - a);
  let sum = 0, sumSq = 0, tauStar = -1;
  for (let i = 0; i < sorted.length; i++) {
    sum += sorted[i]; sumSq += sorted[i] ** 2;
    const n = i + 1, mu = sum / n, ss = n * (sumSq / n - mu ** 2);
    const tau = mu - Math.sqrt(Math.max((1 - ss) / n, 0));
    if (tau <= sorted[i]) tauStar = tau;
  }
  const p = scaled.map(x => Math.max(x - tauStar, 0) ** 2), total = p.reduce((a, b) => a + b, 0);
  return p.map(x => f32(x / total));
}
function expertBatch(request, prior) {
  const logs = request.values.map(row => row.map(v => v === null ? null : f32(Math.log1p(v))));
  const columns = [0, 1, 2, 3].map(q => logs.map(row => row[q]).filter(x => x !== null));
  const medians = columns.map(median);
  const contexts = columns.map((a, q) => {
    const fallback = mean(Array.from({ length: 24 }, (_, h) => prior.data[h * 4 + q]));
    const avg = a.length ? f32(mean(a)) : fallback;
    const std = a.length > 1 ? Math.sqrt(mean(a.map(x => f32(x - avg) ** 2))) : 0;
    const recent = median(logs.slice(-24).map(row => row[q]).filter(x => x !== null));
    const previous = median(logs.slice(-48, -24).map(row => row[q]).filter(x => x !== null));
    return [avg, medians[q] ?? fallback, std, 1 - a.length / 336, recent === null || previous === null ? 0 : recent - previous].map(f32);
  });
  const values = [], availability = [], reliability = [];
  for (let h = 0; h < 24; h++) {
    values[h] = []; availability[h] = []; reliability[h] = [];
    for (let q = 0; q < 4; q++) {
      const p = prior.data[h * 4 + q], v = [], m = [], rho = [];
      for (const d of [1, 7, 14]) { const x = logs[336 + h - 24 * d][q]; v.push(x ?? p); m.push(x !== null); rho.push(x === null ? 0 : 1); }
      for (const days of [7, 14]) {
        const a = Array.from({ length: days }, (_, d) => logs[336 + h - 24 * (d + 1)][q]).filter(x => x !== null);
        v.push(median(a) ?? p); m.push(a.length > 0); rho.push(f32(a.length / days));
      }
      const trend = m[1] && m[2];
      v.push(trend ? f32(v[1] + clamp(f32(v[1] - v[2]), -0.5, 0.5)) : p); m.push(trend); rho.push(trend ? 1 : 0);
      v.push(medians[q] ?? p); m.push(columns[q].length > 0); rho.push(f32(columns[q].length / 336));
      v.push(p); m.push(true); rho.push(1);
      values[h][q] = v; availability[h][q] = m; reliability[h][q] = rho;
    }
  }
  return { values, availability, reliability, contexts };
}
const rawPrediction = (log) => f32(Math.max(1e-4, Math.expm1(log)));
const grid = () => Array.from({ length: 24 }, () => Array(4).fill(0));
export function forecast(request, bundle, onProgress = () => {}) {
  const members = []; let maxUnavailableWeight = 0, maxEnvelopeViolation = 0;
  for (const member of bundle.members) {
    const t = member.tensors, batch = expertBatch(request, t.prior_log);
    const d = member.selected_config.token_dim, bound = member.selected_config.residual_bound;
    const rawBeta = t.reliability_beta_raw.data[0];
    const beta = Math.max(rawBeta, 0) + Math.log1p(Math.exp(-Math.abs(rawBeta)));
    const prediction = grid(), baseline = grid(), residual = grid(), entropy = grid(), lower = grid(), upper = grid();
    const attention = Array.from({ length: 24 }, () => Array(4));
    for (let h = 0; h < 24; h++) for (let q = 0; q < 4; q++) {
      const hEmbedding = rowOf(t['horizon_embedding.weight'], h), qEmbedding = rowOf(t['metric_embedding.weight'], q), context = batch.contexts[q];
      const query = network([...hEmbedding, ...qEmbedding, ...context], 'query_network', t);
      const available = [], scores = [];
      for (let j = 0; j < 8; j++) {
        if (!batch.availability[h][q][j]) continue;
        const key = network([...rowOf(t['expert_embedding.weight'], j), batch.values[h][q][j], 1, t.expert_distance.data[j], batch.reliability[h][q][j]], 'key_network', t);
        let dot = 0; for (let k = 0; k < d; k++) dot += query[k] * key[k];
        available.push(j); scores.push(f32(dot / Math.sqrt(d) + beta * Math.log(Math.max(batch.reliability[h][q][j], 1e-6))));
      }
      const allocation = entmax15(scores), alpha = Array(8).fill(0);
      allocation.forEach((x, i) => { alpha[available[i]] = x; });
      const b = f32(alpha.reduce((sum, x, j) => sum + x * batch.values[h][q][j], 0));
      const H = f32(-alpha.reduce((sum, x) => sum + x * Math.log(Math.max(x, 1e-12)), 0));
      const delta = f32(bound * Math.tanh(network([b, ...context, ...hEmbedding, ...qEmbedding, H], 'residual_network', t)[0]));
      const log = f32(b + delta);
      const v = available.map(j => batch.values[h][q][j]);
      prediction[h][q] = rawPrediction(log); lower[h][q] = rawPrediction(Math.min(...v) - bound); upper[h][q] = rawPrediction(Math.max(...v) + bound);
      if (!Number.isFinite(prediction[h][q])) fail('numerical_range');
      baseline[h][q] = b; residual[h][q] = delta; entropy[h][q] = H; attention[h][q] = alpha;
      const violation = Math.max(lower[h][q] - prediction[h][q], prediction[h][q] - upper[h][q], 0);
      maxEnvelopeViolation = Math.max(maxEnvelopeViolation, violation);
      if (violation > 1e-5 * Math.max(1, prediction[h][q])) fail('numerical_range');
      alpha.forEach((a, j) => { if (!batch.availability[h][q][j]) maxUnavailableWeight = Math.max(maxUnavailableWeight, a); });
    }
    members.push({ seed: member.seed, checkpoint_sha256: member.checkpoint_sha256, selected_config: member.selected_config,
      expert_values_log: batch.values, availability: batch.availability, reliability: batch.reliability, context: batch.contexts,
      attention, baseline_log: baseline, residual_log: residual, entropy, prediction, lower_envelope: lower, upper_envelope: upper });
    onProgress(members.length, bundle.members.length);
  }
  const prediction = grid();
  for (let h = 0; h < 24; h++) for (let q = 0; q < 4; q++) prediction[h][q] = f32(members.reduce((sum, m) => sum + m.prediction[h][q], 0) / members.length);
  return { prediction, forecastTimes: Array.from({ length: 24 }, (_, h) => request.times.at(-1) + (h + 1) * 3600000), members,
    checks: { max_unavailable_weight: maxUnavailableWeight, max_envelope_violation: maxEnvelopeViolation },
    model: { repo_id: bundle.manifest.repo_id, revision: bundle.manifest.revision, weights_sha256: bundle.manifest.weights.sha256, seeds: members.map(m => m.seed) } };
}
export function formatTimestamp(time) {
  const d = new Date(time), pad = x => String(x).padStart(2, '0');
  return `${d.getUTCFullYear()}/${pad(d.getUTCMonth() + 1)}/${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
}
export function forecastCSV(request, result) {
  const quote = x => `"${String(x).replaceAll('"', '""')}"`;
  return '\uFEFF' + [HEADER, ...result.prediction.map((row, i) => [formatTimestamp(result.forecastTimes[i]), request.cell, ...row])].map(row => row.map(quote).join(',')).join('\r\n') + '\r\n';
}
