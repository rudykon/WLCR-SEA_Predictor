import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parseCSV, parseRequest, corruptRequest, decodeModel, forecast, forecastCSV, entmax15, REVISION } from '../docs/javascripts/local-demo-core.mjs';

const root = new URL('../', import.meta.url);
const sample = await readFile(new URL('docs/assets/local-demo/sample.csv', root), 'utf8');
const manifest = JSON.parse(await readFile(new URL('docs/assets/local-demo/model.json', root), 'utf8'));
const bytes = await readFile(new URL('docs/assets/local-demo/weights.bin', root));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const bundle = await decodeModel(manifest, buffer);
const reference = JSON.parse(await readFile(new URL('tests/fixtures/browser_demo_reference.json', root), 'utf8'));
const quote = x => `"${String(x).replaceAll('"', '""')}"`;
const csvOf = rows => rows.map(row => row.map(quote).join(',')).join('\r\n');
function sourceCSV(source) {
  const rows = parseCSV(sample);
  for (let i = 1; i < rows.length; i++) for (let q = 0; q < 4; q++) {
    if (source === 'zeros') rows[i][q + 2] = '0';
    if (source === 'sparse' && i !== 336) rows[i][q + 2] = 'NIL';
    if (source === 'gapped' && (i - 1 + q) % 5 === 0) rows[i][q + 2] = '';
  }
  return csvOf(rows);
}
const close = (a, b) => assert.ok(Math.abs(a - b) <= 1e-4 + 5e-5 * Math.abs(b), `${a} differs from ${b}`);

test('public browser assets match the pinned model release and reject corrupt bytes', async () => {
  assert.equal(manifest.revision, REVISION);
  assert.equal(reference.model_revision, REVISION);
  assert.equal(reference.weights_sha256, manifest.weights.sha256);
  const changed = buffer.slice(0); new Uint8Array(changed)[0] ^= 1;
  await assert.rejects(decodeModel(manifest, changed), /model_checksum/);
  const incomplete = { ...manifest, members: manifest.members.slice(0, 4) };
  await assert.rejects(decodeModel(incomplete, buffer), /invalid_model/);
});

test('all eight browser forecasts agree with independently calculated Python references', async t => {
  let maxAbsoluteError = 0, maxRelativeError = 0;
  for (const c of reference.cases) await t.test(c.name, async () => {
    const parsed = parseRequest(sourceCSV(c.source));
    const request = await corruptRequest(parsed, c.scenario, c.rate);
    assert.equal(request.observed, c.observed, 'the SHA-256-based corruption mask must match Python');
    const result = forecast(request, bundle);
    assert.equal(result.prediction.length, 24);
    for (let h = 0; h < 24; h++) for (let q = 0; q < 4; q++) {
      const expected = c.prediction[h][q], actual = result.prediction[h][q];
      close(actual, expected);
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(actual - expected));
      maxRelativeError = Math.max(maxRelativeError, Math.abs(actual - expected) / Math.max(1e-4, Math.abs(expected)));
      close(actual, result.members.reduce((sum, m) => sum + m.prediction[h][q], 0) / 5);
      for (const member of result.members) {
        assert.ok(Number.isFinite(member.prediction[h][q]) && member.prediction[h][q] >= Math.fround(1e-4));
        assert.ok(Math.abs(member.residual_log[h][q]) <= member.selected_config.residual_bound);
        close(member.attention[h][q].reduce((a, b) => a + b, 0), 1);
        member.attention[h][q].forEach((weight, j) => { if (!member.availability[h][q][j]) assert.equal(weight, 0); });
      }
    }
    assert.equal(result.checks.max_unavailable_weight, 0);
    assert.equal(result.checks.max_envelope_violation, 0);
    const rows = parseCSV(forecastCSV(request, result));
    assert.equal(rows.length, 25);
    assert.ok(rows.every(row => row.length === 6));
  });
  t.diagnostic(`Max absolute Python/browser error: ${maxAbsoluteError}; max relative error: ${maxRelativeError}`);
});

test('request parsing enforces one contiguous, finite 336-hour cell window', () => {
  const rows = parseCSV(sample), mutate = action => { const copy = rows.map(row => [...row]); action(copy); return csvOf(copy); };
  assert.throws(() => parseRequest(mutate(rows => rows.pop())), /wrong_row_count/);
  assert.throws(() => parseRequest(mutate(rows => rows[2][1] = 'another-cell')), /mixed_cells/);
  assert.throws(() => parseRequest(mutate(rows => rows[2][0] = rows[1][0])), /non_contiguous_timestamps/);
  assert.throws(() => parseRequest(mutate(rows => rows[1][0] = '2026/02/30 00:00')), /invalid_timestamp/);
  for (const value of ['-1', 'Infinity', 'NaN', '1e400', '12cats']) assert.throws(() => parseRequest(mutate(rows => rows[1][2] = value)), /invalid_metric/);
  assert.throws(() => parseRequest(mutate(rows => rows.slice(1).forEach(row => row[2] = 'NIL'))), /indicator_unobserved/);
  assert.throws(() => parseCSV('a,"unclosed'), /invalid_csv/);
  assert.throws(() => parseCSV('a,"closed"extra'), /invalid_csv/);
  const quoted = mutate(rows => rows.slice(1).forEach(row => row[1] = 'Cell, "east"'));
  assert.equal(parseRequest(quoted).cell, 'Cell, "east"');
  assert.deepEqual(parseRequest(sample.replace(/^\uFEFF/, '')).values, parseRequest(sample).values);
});

test('corruption is deterministic, non-mutating, and rejects total indicator removal', async () => {
  const request = parseRequest(sample), original = JSON.stringify(request);
  const first = await corruptRequest(request, 'asynchronous', 0.5);
  assert.deepEqual(first, await corruptRequest(request, 'asynchronous', 0.5));
  assert.equal(JSON.stringify(request), original);
  await assert.rejects(corruptRequest(request, 'mcar', 0.9), /invalid_missing_rate/);
  await assert.rejects(corruptRequest(request, 'unregistered', 0.5), /unknown_scenario/);
  await assert.rejects(corruptRequest(parseRequest(sourceCSV('sparse')), 'recent_tail', 0.5), /indicator_removed/);
});

test('model computation ignores wrapper cell identity and routing has exact zeros', async () => {
  const request = await corruptRequest(parseRequest(sample), 'block', 0.5);
  assert.deepEqual(forecast(request, bundle).prediction, forecast({ ...request, cell: 'wrapper-only-change' }, bundle).prediction);
  assert.deepEqual(entmax15([10]), [1]);
  const p = entmax15([10, -10]); assert.equal(p[0], 1); assert.equal(p[1], 0);
});

test('routing inspection follows the selected hour and preserves the ensemble bounds', async () => {
  const { routingAt, ensembleEnvelope } = await import('../docs/javascripts/local-demo-view.mjs');
  const request = await corruptRequest(parseRequest(sample), 'recent_tail', 0.5);
  const result = forecast(request, bundle);
  const first = routingAt(result, 2, 0), last = routingAt(result, 2, 23);
  assert.notDeepEqual(first.map(r => r.weight), last.map(r => r.weight), 'a future-hour change must inspect that hour, not repeat the 24-hour mean');
  for (const h of [0, 23]) {
    const rows = routingAt(result, 2, h);
    close(rows.reduce((sum, row) => sum + row.weight, 0), 1);
    rows.forEach((row, j) => {
      close(row.weight, result.members.reduce((sum, m) => sum + m.attention[h][2][j], 0) / 5);
      if (!row.available) { assert.equal(row.value, null); assert.equal(row.weight, 0); }
    });
  }
  const bounds = ensembleEnvelope(result);
  for (let h = 0; h < 24; h++) for (let q = 0; q < 4; q++) {
    assert.ok(bounds.lower[h][q] <= result.prediction[h][q] + 1e-4);
    assert.ok(bounds.upper[h][q] >= result.prediction[h][q] - 1e-4);
  }
  assert.throws(() => routingAt(result, 2, 24), RangeError);
});
