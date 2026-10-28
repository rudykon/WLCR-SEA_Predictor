import { parseRequest, corruptRequest, decodeModel, forecast, sha256 } from './local-demo-core.mjs';

let bundlePromise;
async function loadModel() {
  if (!bundlePromise) bundlePromise = (async () => {
    const base = new URL('../assets/local-demo/', import.meta.url);
    const metadata = await fetch(new URL('model.json', base), { cache: 'force-cache' });
    if (!metadata.ok) throw new Error('model_download');
    const manifest = await metadata.json();
    // The runtime fetches only these same-origin static assets, never input data.
    if (manifest.weights?.file !== 'weights.bin') throw new Error('invalid_model');
    const response = await fetch(new URL('weights.bin', base), { cache: 'force-cache' });
    if (!response.ok) throw new Error('model_download');
    return decodeModel(manifest, await response.arrayBuffer());
  })().catch(error => {
    bundlePromise = null;
    if (['invalid_model', 'model_checksum'].includes(error.message)) throw error;
    throw new Error('model_download');
  });
  return bundlePromise;
}
let busy = false;
self.onmessage = async ({ data }) => {
  if (busy) return;
  busy = true;
  try {
    if (!self.crypto?.subtle) throw new Error('secure_context');
    const original = parseRequest(data.csv);
    const request = await corruptRequest(original, data.scenario, data.rate);
    self.postMessage({ type: 'status', stage: 'loading' });
    const bundle = await loadModel();
    self.postMessage({ type: 'status', stage: 'running', member: 0 });
    const start = performance.now();
    const result = forecast(request, bundle, (member, total) => self.postMessage({ type: 'status', stage: 'running', member, total }));
    const elapsedMs = performance.now() - start;
    const inputTextSHA256 = await sha256(new TextEncoder().encode(data.csv.replace(/^\uFEFF/, '')));
    self.postMessage({ type: 'result', request, result, elapsedMs, inputTextSHA256 });
  } catch (error) {
    self.postMessage({ type: 'error', code: error.message });
  } finally { busy = false; }
};
