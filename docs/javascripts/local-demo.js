/* Progressive enhancement for the existing bilingual MkDocs site. */
(() => {
  const root = document.querySelector('[data-local-demo]');
  if (!root) return;
  const scriptURL = new URL(document.currentScript.src);
  const zh = root.dataset.lang === 'zh';
  const t = zh ? {
    badge: '在你的电脑上运行', intro: '选择内置样例，或打开一个本地 CSV。五个公开模型在浏览器后台计算，文件内容不会上传。',
    source: '输入数据', sample: '内置合成样例', file: '选择本地 CSV', choose: 'CSV 文件', fileHint: 'UTF-8，336 行，单个小区；缺失值使用 NIL 或空白。上限 5 MB。',
    pattern: '模拟缺失', patterns: ['保持原始观测', '随机整小时缺失', '连续区块缺失', '最近时段中断', '指标异步缺失'], rate: '设定移除率',
    run: '在本机运行预测', cancel: '取消', sampleDownload: '下载样例 CSV', csv: '下载预测 CSV', audit: '下载审计 JSON',
    ready: '就绪，点击运行即可开始。', loading: '加载并校验公开模型…', running: n => `正在本机计算：${n} / 5 个模型`, done: '预测完成，计算发生在当前浏览器。', cancelled: '已取消。你可以调整输入后重新运行。',
    history: '历史窗口', hours: '336 小时', observed: '可用观测', compute: '计算设备', device: '本机 CPU', latency: '本机推理耗时', ensemble: '模型集成', five: '5 个公开模型',
    chart: '历史与未来 24 小时', observedLegend: '历史观测', forecastLegend: '五模型预测', recent: '图中显示最近 72 小时及未来 24 小时；历史曲线的空缺表示缺失观测。',
    metrics: ['上行活跃用户数', '下行活跃用户数', '下行已用 PRB 数', '上行已用 PRB 数'],
    routing: '哪些参考参与了预测', routingHelp: '展示所选指标在 24 个预测小时、五个模型上的平均分配。下载的审计记录保留每个成员的独立计算。',
    experts: ['前一天', '前一周', '两周前', '7 日同小时中位数', '14 日同小时中位数', '限幅周趋势', '窗口中位数', '冻结训练先验'],
    table: '未来 24 小时', time: '时间', tableMetrics: ['上行用户', '下行用户', '下行 PRB', '上行 PRB'], resultHint: '运行后可查看预测表和专家分配，并下载结果。',
    note: '使用公开的种子 42–46 检查点。浏览器数值与 Python 后端可能存在微小浮点差异。模型在页面内复用；数据只保存在本页内存中。',
    removed: (a, b, p) => `原始观测 ${a} 项，本次额外移除 ${b} 项（${p}%）。`,
    errors: {
      empty_csv: 'CSV 为空。', invalid_csv: 'CSV 格式不正确，请使用样例的六列格式。', invalid_header: 'CSV 表头不匹配，请参考下载的样例。',
      wrong_row_count: '需要恰好 336 行小时记录。', invalid_columns: '每一行必须包含六列。', empty_cell: '小区名称不能为空。', mixed_cells: '全部记录必须属于同一小区。',
      invalid_timestamp: '时间必须使用有效的 YYYY/MM/DD HH:MM 格式。', non_contiguous_timestamps: '记录必须严格按小时连续排列。',
      invalid_metric: '指标必须是非负有限数值、NIL 或空白。', indicator_unobserved: '每项指标至少需要一个有效观测。', indicator_removed: '当前设置移除了某项指标的全部观测，请降低移除率。',
      invalid_missing_rate: '移除率须为 0–80%。', missing_file: '请先选择 CSV 文件。', file_too_large: '文件超过 5 MB。', invalid_extension: '请选择 .csv 文件。', invalid_encoding: '请使用 UTF-8 编码的 CSV。',
      model_download: '模型资源加载失败，请检查网络后重试。', model_checksum: '模型资源校验失败，请刷新页面后重试。', invalid_model: '模型资源格式不匹配，请刷新页面。',
      secure_context: '请通过 HTTPS 或 localhost 打开本页。', worker: '浏览器后台线程未能运行，请换用较新的浏览器或刷新重试。', numerical_range: '输入数值超出可计算范围，请检查原始数值。', unknown: '运行失败，请检查输入后重试。', sample: '样例读取失败，你仍可选择本地 CSV。'
    }
  } : {
    badge: 'Runs on your computer', intro: 'Use the synthetic sample or open a local CSV. Five public models compute in a browser worker. Your file contents are never uploaded.',
    source: 'Input data', sample: 'Built-in synthetic sample', file: 'Choose a local CSV', choose: 'CSV file', fileHint: 'UTF-8, 336 rows, one cell; use NIL or a blank for missing values. Up to 5 MB.',
    pattern: 'Simulate missing data', patterns: ['Keep original observations', 'Random whole-hour loss', 'Contiguous block loss', 'Recent-tail outage', 'Asynchronous indicator loss'], rate: 'Requested removal',
    run: 'Run forecast on this device', cancel: 'Cancel', sampleDownload: 'Download sample CSV', csv: 'Download forecast CSV', audit: 'Download audit JSON',
    ready: 'Ready. Run the forecast to begin.', loading: 'Loading and verifying the public models…', running: n => `Computing on this device: ${n} / 5 models`, done: 'Forecast complete. Computed in this browser.', cancelled: 'Cancelled. Adjust the input and run again.',
    history: 'History window', hours: '336 hours', observed: 'Observed values', compute: 'Compute device', device: 'Your CPU', latency: 'Local inference time', ensemble: 'Ensemble', five: '5 public models',
    chart: 'History and the next 24 hours', observedLegend: 'Observed history', forecastLegend: 'Five-model forecast', recent: 'The chart shows the last 72 hours and next 24 hours. Gaps in history indicate missing observations.',
    metrics: ['UL active users', 'DL active users', 'Used DL PRBs', 'Used UL PRBs'],
    routing: 'Which references contributed', routingHelp: 'Mean allocation for the selected indicator across 24 forecast hours and five models. The audit download keeps each member’s independent computation.',
    experts: ['Previous day', 'Previous week', 'Two weeks earlier', '7-day same-hour median', '14-day same-hour median', 'Limited weekly trend', 'Window median', 'Frozen training prior'],
    table: 'Next 24 hours', time: 'Time', tableMetrics: ['UL users', 'DL users', 'DL PRBs', 'UL PRBs'], resultHint: 'Run a forecast to inspect the table and expert allocation, and download the results.',
    note: 'Uses the public seed 42–46 checkpoints. Browser and Python results may differ slightly due to floating-point arithmetic. Models are reused within the page; request data stays in page memory.',
    removed: (a, b, p) => `${a} original observations; ${b} additionally removed (${p}%).`,
    errors: {
      empty_csv: 'The CSV is empty.', invalid_csv: 'Invalid CSV. Use the sample’s six-column format.', invalid_header: 'The header does not match the sample CSV.',
      wrong_row_count: 'Provide exactly 336 hourly records.', invalid_columns: 'Every row must contain six columns.', empty_cell: 'Cell names cannot be empty.', mixed_cells: 'All rows must belong to one cell.',
      invalid_timestamp: 'Use valid YYYY/MM/DD HH:MM timestamps.', non_contiguous_timestamps: 'Records must be strictly hourly and contiguous.',
      invalid_metric: 'Use finite non-negative numbers, NIL, or blanks.', indicator_unobserved: 'Each indicator needs at least one observed value.', indicator_removed: 'This setting removes every observation of an indicator. Lower the removal rate.',
      invalid_missing_rate: 'The removal rate must be between 0 and 80%.', missing_file: 'Choose a CSV file first.', file_too_large: 'The file exceeds 5 MB.', invalid_extension: 'Choose a .csv file.', invalid_encoding: 'Use a UTF-8 CSV file.',
      model_download: 'Could not load the models. Check your connection and try again.', model_checksum: 'Model verification failed. Refresh the page and retry.', invalid_model: 'Model format mismatch. Refresh the page.',
      secure_context: 'Open this page over HTTPS or localhost.', worker: 'The browser worker could not run. Use a recent browser or refresh and retry.', numerical_range: 'The input exceeds the supported numeric range. Check its values.', unknown: 'The forecast failed. Check your input and retry.', sample: 'The sample could not load. You can still choose a local CSV.'
    }
  };
  const scenarios = ['none', 'mcar', 'block', 'recent_tail', 'asynchronous'];
  root.innerHTML = `
    <div class="local-demo-intro"><span class="local-demo-badge">${t.badge}</span><p>${t.intro}</p></div>
    <section class="local-demo-controls" aria-label="${t.source}">
      <div class="local-demo-fields">
        <label>${t.source}<select id="local-source"><option value="sample">${t.sample}</option><option value="file">${t.file}</option></select></label>
        <label>${t.pattern}<select id="local-pattern">${scenarios.map((s, i) => `<option value="${s}">${t.patterns[i]}</option>`).join('')}</select></label>
        <label class="local-demo-rate">${t.rate} <output id="local-rate-value">30%</output><input id="local-rate" type="range" min="0" max="80" step="5" value="30" disabled></label>
      </div>
      <div id="local-upload" hidden><label>${t.choose}<input id="local-file" type="file" accept=".csv,text/csv" aria-describedby="local-file-hint"></label><p id="local-file-hint" class="local-demo-help">${t.fileHint}</p></div>
      <div class="local-demo-actions"><button type="button" id="local-run" class="md-button md-button--primary" disabled>${t.run}</button><button type="button" id="local-cancel" class="md-button" hidden>${t.cancel}</button><button type="button" id="local-sample" class="md-button" disabled>${t.sampleDownload}</button></div>
      <p id="local-status" role="status" aria-live="polite">${t.ready}</p><p id="local-error" role="alert" hidden></p>
    </section>
    <div class="local-demo-stats"><div><span>${t.history}</span><strong>${t.hours}</strong></div><div><span>${t.observed}</span><strong id="local-observed">—</strong></div><div><span id="local-compute-label">${t.compute}</span><strong id="local-compute">${t.device}</strong></div><div><span>${t.ensemble}</span><strong>${t.five}</strong></div></div>
    <section class="local-demo-panel"><h2>${t.chart}</h2><div class="local-demo-metrics" role="group" aria-label="${t.chart}">${t.metrics.map((s, i) => `<button type="button" data-metric="${i}" aria-pressed="${i === 2}">${s}</button>`).join('')}</div><div class="local-demo-legend"><span class="local-demo-history">${t.observedLegend}</span><span class="local-demo-forecast">${t.forecastLegend}</span></div><div id="local-chart" aria-label="${t.chart}"></div><p class="local-demo-help">${t.recent}</p><p id="local-removal" class="local-demo-help"></p></section>
    <section class="local-demo-panel" id="local-results" hidden><div class="local-demo-result-heading"><h2>${t.table}</h2><div class="local-demo-actions"><button type="button" id="local-csv" class="md-button" disabled>${t.csv}</button><button type="button" id="local-audit" class="md-button" disabled>${t.audit}</button></div></div><p id="local-window"></p><div class="local-demo-table"><table><thead><tr><th>${t.time}</th>${t.tableMetrics.map(x => `<th>${x}</th>`).join('')}</tr></thead><tbody id="local-tbody"></tbody></table></div><h3>${t.routing}</h3><p class="local-demo-help">${t.routingHelp}</p><div id="local-routing" role="list"></div></section>
    <p class="local-demo-note">${t.note}</p>`;
  const el = id => root.querySelector(`#local-${id}`);
  const number = new Intl.NumberFormat(zh ? 'zh-CN' : 'en', { maximumFractionDigits: 3 });
  let core, sampleText, csvText, request, output, worker, busy = false, revision = 0, metric = 2;
  const showError = code => { el('error').textContent = t.errors[code] || t.errors.unknown; el('error').hidden = false; root.dataset.state = 'error'; };
  function setBusy(value) {
    busy = value; el('cancel').hidden = !value;
    for (const id of ['source', 'pattern', 'file', 'run']) el(id).disabled = value;
    el('rate').disabled = value || el('pattern').value === 'none';
    el('run').disabled = value || !request;
    if (value) { el('error').hidden = true; root.dataset.state = 'running'; }
  }
  function invalidate() {
    output = undefined; el('results').hidden = true; el('csv').disabled = true; el('audit').disabled = true;
    el('compute-label').textContent = t.compute; el('compute').textContent = t.device;
    el('removal').textContent = ''; el('status').textContent = t.ready; el('error').hidden = true; root.dataset.state = 'idle';
    el('observed').textContent = request ? `${request.values.flat().filter(x => x !== null).length} / 1,344` : '—';
    drawChart();
  }
  function download(name, contents, type) {
    const url = URL.createObjectURL(new Blob([contents], { type }));
    const a = document.createElement('a'); a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function drawChart() {
    const box = el('chart');
    if (!request) { box.replaceChildren(); return; }
    const history = output?.request.values || request.values, result = output?.result;
    const width = Math.max(320, box.clientWidth || 700), height = 280, left = 56, right = width - 18, top = 18, bottom = height - 44;
    const first = 264, last = 359, points = history.slice(first).map(row => row[metric]).filter(x => x !== null);
    if (result) points.push(...result.prediction.map(row => row[metric]));
    const max = Math.max(1, ...points) * 1.12, x = i => left + (i - first) / (last - first) * (right - left), y = v => bottom - v / max * (bottom - top);
    const ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${t.metrics[metric]}: ${t.chart}`);
    const add = (tag, attrs, text) => { const e = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v)); if (text !== undefined) e.textContent = text; svg.append(e); return e; };
    const divider = x(335.5);
    add('rect', { x: divider, y: top, width: right - divider, height: bottom - top, fill: 'var(--wl-soft)' });
    for (let i = 0; i <= 4; i++) { const v = max * i / 4; add('line', { x1: left, y1: y(v), x2: right, y2: y(v), stroke: 'var(--wl-border)', 'stroke-width': 1 }); add('text', { x: left - 7, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--wl-muted)', 'font-size': 11 }, number.format(v)); }
    add('line', { x1: divider, x2: divider, y1: top, y2: bottom, stroke: 'var(--wl-muted)', 'stroke-dasharray': '3 4' });
    const path = (rows, offset, column) => { let d = '', open = false; for (let i = 0; i < rows.length; i++) { const v = rows[i][column]; if (v === null) { open = false; continue; } d += `${open ? 'L' : 'M'}${x(offset + i).toFixed(2)},${y(v).toFixed(2)} `; open = true; } return d; };
    add('path', { d: path(history.slice(first), first, metric), fill: 'none', stroke: 'var(--wl-blue)', 'stroke-width': 2.2, 'stroke-linejoin': 'round' });
    // Show isolated observations too; a one-point SVG path has no visible stroke.
    for (let i = first; i < 336; i++) {
      const value = history[i][metric];
      if (value !== null && (i === first || i === 335 || history[i - 1][metric] === null || history[i + 1][metric] === null))
        add('circle', { cx: x(i), cy: y(value), r: 1.7, fill: 'var(--wl-blue)' });
    }
    if (result) {
      add('path', { d: path(result.prediction, 336, metric), fill: 'none', stroke: 'var(--wl-teal)', 'stroke-width': 2.4, 'stroke-dasharray': '5 3' });
      for (let h = 0; h < 24; h += 3) add('circle', { cx: x(336 + h), cy: y(result.prediction[h][metric]), r: 2.5, fill: 'var(--wl-surface)', stroke: 'var(--wl-teal)', 'stroke-width': 1.5 });
    }
    const ticks = width < 480 ? [264, 312, 359] : [264, 288, 312, 335, 359];
    for (const i of ticks) {
      const time = i < 336 ? request.times[i] : request.times.at(-1) + (i - 335) * 3600000;
      const text = core.formatTimestamp(time).slice(5, 16);
      add('text', { x: x(i), y: bottom + 23, 'text-anchor': i === 264 ? 'start' : i === 359 ? 'end' : 'middle', fill: 'var(--wl-muted)', 'font-size': 11 }, text);
    }
    box.replaceChildren(svg);
  }
  function drawRouting() {
    if (!output) return;
    el('routing').replaceChildren();
    for (let j = 0; j < 8; j++) {
      const mass = output.result.members.reduce((sum, m) => sum + m.attention.reduce((s, row) => s + row[metric][j], 0), 0) / 120;
      const row = document.createElement('div'); row.className = 'local-demo-routing-row'; row.setAttribute('role', 'listitem');
      const label = document.createElement('span'); label.textContent = t.experts[j];
      const meter = document.createElement('meter'); meter.min = 0; meter.max = 1; meter.value = mass; meter.setAttribute('aria-label', t.experts[j]);
      const value = document.createElement('span'); value.textContent = `${(mass * 100).toFixed(1)}%`;
      row.append(label, meter, value); el('routing').append(row);
    }
  }
  function showResult(data) {
    output = data; root.dataset.state = 'done'; el('status').textContent = t.done;
    el('observed').textContent = `${data.request.observed} / 1,344`; el('compute-label').textContent = t.latency; el('compute').textContent = `${number.format(data.elapsedMs)} ms`;
    el('removal').textContent = t.removed(data.request.originalObserved, data.request.newlyRemoved, (data.request.removedFraction * 100).toFixed(1));
    el('window').textContent = `${request.cell} · ${core.formatTimestamp(data.result.forecastTimes[0])} – ${core.formatTimestamp(data.result.forecastTimes[23])}`;
    el('tbody').replaceChildren();
    data.result.prediction.forEach((row, i) => { const tr = document.createElement('tr'); for (const text of [core.formatTimestamp(data.result.forecastTimes[i]), ...row.map(x => number.format(x))]) { const td = document.createElement('td'); td.textContent = text; tr.append(td); } el('tbody').append(tr); });
    el('results').hidden = false; el('csv').disabled = false; el('audit').disabled = false; drawChart(); drawRouting();
  }
  function getWorker() {
    if (!worker) {
      worker = new Worker(new URL('local-demo-worker.mjs', scriptURL), { type: 'module' });
      worker.onmessage = ({ data }) => {
        if (data.type === 'status') el('status').textContent = data.stage === 'loading' ? t.loading : t.running(data.member || 0);
        else if (data.type === 'error') { setBusy(false); showError(data.code); el('status').textContent = ''; }
        else if (data.type === 'result') { setBusy(false); showResult(data); }
      };
      worker.onerror = () => { worker?.terminate(); worker = undefined; setBusy(false); showError('worker'); el('status').textContent = ''; };
    }
    return worker;
  }
  async function selectInput() {
    const version = ++revision; request = undefined; csvText = undefined; invalidate();
    const fromFile = el('source').value === 'file'; el('upload').hidden = !fromFile; el('run').disabled = true;
    try {
      let text;
      if (!fromFile) text = sampleText;
      else {
        const file = el('file').files[0]; if (!file) return;
        if (!file.name.toLowerCase().endsWith('.csv')) throw new Error('invalid_extension');
        if (file.size > 5 * 1024 * 1024) throw new Error('file_too_large');
        try { text = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer()); } catch { throw new Error('invalid_encoding'); }
      }
      if (version !== revision || text === undefined) return;
      const parsed = core.parseRequest(text); csvText = text; request = parsed; invalidate(); el('run').disabled = false;
    } catch (e) { if (version === revision) showError(e.message); }
  }
  root.querySelectorAll('[data-metric]').forEach(button => button.addEventListener('click', () => { metric = Number(button.dataset.metric); root.querySelectorAll('[data-metric]').forEach(b => b.setAttribute('aria-pressed', String(b === button))); drawChart(); drawRouting(); }));
  el('source').addEventListener('change', selectInput); el('file').addEventListener('change', selectInput);
  el('pattern').addEventListener('change', () => { invalidate(); el('rate').disabled = el('pattern').value === 'none'; });
  el('rate').addEventListener('input', () => { el('rate-value').textContent = `${el('rate').value}%`; invalidate(); });
  el('run').addEventListener('click', () => {
    if (!request || busy) return;
    invalidate(); setBusy(true); el('status').textContent = t.loading;
    try { getWorker().postMessage({ csv: csvText, scenario: el('pattern').value, rate: Number(el('rate').value) / 100 }); }
    catch { setBusy(false); showError('worker'); }
  });
  el('cancel').addEventListener('click', () => { worker?.terminate(); worker = undefined; setBusy(false); invalidate(); el('status').textContent = t.cancelled; });
  el('sample').addEventListener('click', () => download('synthetic_traffic.csv', sampleText, 'text/csv;charset=utf-8'));
  el('csv').addEventListener('click', () => { if (output) download('wlcr-sea-forecast.csv', core.forecastCSV(request, output.result), 'text/csv;charset=utf-8'); });
  el('audit').addEventListener('click', () => {
    if (!output) return;
    const audit = { schema: 'wlcr-sea-browser-audit/v1', runtime: 'wlcr-sea-browser/1', executed_on: 'visitor_device_cpu',
      input_text_sha256: output.inputTextSHA256, input_hash_encoding: 'UTF-8 decoded text with BOM removed',
      model: output.result.model, request: output.request, forecast_times: output.result.forecastTimes.map(core.formatTimestamp),
      prediction: output.result.prediction, members: output.result.members, checks: output.result.checks,
      inference_ms: output.elapsedMs,
      numerical_implementation: 'Public float32 tensors; JavaScript accumulation and erf GELU (not bitwise PyTorch identity).' };
    download('wlcr-sea-audit.json', JSON.stringify(audit, null, 2), 'application/json');
  });
  let oldWidth = 0;
  new ResizeObserver(entries => { const width = Math.round(entries[0].contentRect.width); if (width !== oldWidth) { oldWidth = width; if (core) drawChart(); } }).observe(el('chart'));
  window.addEventListener('pagehide', () => worker?.terminate());
  (async () => {
    try {
      core = await import(new URL('local-demo-core.mjs', scriptURL));
      const response = await fetch(new URL('../assets/local-demo/sample.csv', scriptURL));
      if (!response.ok) throw new Error('sample');
      sampleText = await response.text(); el('sample').disabled = false;
      if (el('source').value === 'sample') await selectInput();
    } catch (e) { showError(e.message === 'sample' ? 'sample' : 'worker'); }
  })();
})();
