/* Browser-local interface, aligned with the public Gradio Demo. */
(() => {
  const root = document.querySelector('[data-local-demo]');
  if (!root) return;
  const scriptURL = new URL(document.currentScript.src);
  const projectURL = new URL('../', scriptURL);
  const zh = root.dataset.lang === 'zh';
  const t = zh ? {
    meta: '五模型集成 · 公开检查点 · 本机 CPU 推理', subtitle: '单个小区 · 336 小时历史 → 未来 24 小时预测',
    github: 'GitHub 仓库', website: '项目网站', languages: '界面语言',
    reset: '恢复内置样例', choose: '选择 336 小时 CSV', drop: '拖放 CSV 到这里，或点击选择', builtin: '内置合成样例', localFile: '本地文件', emptyFile: '尚未选择 CSV',
    privacy: '文件留在浏览器中，使用本机 CPU 计算。', fileHint: 'UTF-8 · 336 行 · NIL 或空白表示缺失 · 最大 5 MB',
    sampleDownload: '下载样例 CSV', clear: '清除文件', pattern: '缺失模式', patterns: ['完整历史', '随机整小时缺失', '连续区块缺失', '最近时段中断', '指标异步缺失'], rate: '追加缺失率',
    run: '运行预测', cancel: '取消', ready: '选择文件或恢复内置样例后运行预测。', loading: '正在加载并校验公开模型…', running: n => `正在本机计算：${n} / 5 个模型`, done: '预测完成', preview: '内置样例自动预览', cancelled: '已取消，点击运行预测可重新开始。',
    chart: '历史与未来 24 小时预测', history: '历史观测', forecast: '五模型集成', envelope: '审计包络', missing: '缺失位置',
    fullHistory: '336 小时历史', recentHistory: '最近 72 小时', range: '历史显示范围', chartHelp: '阴影为模型的可检查预测边界；红色短线标记缺失位置。',
    scenario: '场景', configured: '配置移除率', actuallyRemoved: '实际移除率', observed: '最终观测比例', audit: '审计', maskLabel: '掩码', boundLabel: '边界', passed: '通过', failed: '失败',
    runtime: '本机推理', milliseconds: 'ms',
    routing: '集成路由摘要', routingHelp: '展示所选时刻五个成员的平均专家值与平均路由权重，用于观察内部路由倾向。最终集成是成员预测的平均，不能由这两组均值直接重构。',
    indicator: '流量指标', horizon: '未来第几小时', future: h => `未来第 ${h} 小时`,
    metrics: ['上行平均激活用户数', '下行平均激活用户数', '下行平均使用 PRB 数', '上行平均使用 PRB 数'],
    expertValue: '集成平均专家值', meanWeight: '平均路由权重', shortValue: '专家值', shortWeight: '权重', unavailable: '不可用专家',
    experts: ['前一天', '前一周', '前两周', '7 日同小时中位数', '14 日同小时中位数', '限幅周趋势', '窗口中位数', '冻结训练先验'],
    shortExperts: ['前1天', '前1周', '前2周', '中位\n7日', '中位\n14日', '周趋势', '窗口', '先验'],
    advanced: '高级详情', table: '完整 24 × 4 预测', expertTable: '所选时刻的八个季节专家', memberTable: '集成成员',
    time: '时间', tableMetrics: ['上行用户', '下行用户', '下行 PRB', '上行 PRB'],
    expert: '季节专家', available: '可用', yes: '是', no: '否', support: '支持度', seed: '种子', config: '配置', memberPrediction: '所选小时预测', residual: '对数域修正量', bound: '残差上限',
    csv: '下载预测 CSV', auditDownload: '下载审计 JSON',
    downloadHint: '点击“运行预测”后可下载预测与审计记录。自动预览只用于查看结果。', downloadReady: '预测与逐成员审计记录已可下载，文件在浏览器中生成。',
    inputFormat: '输入格式', modelWeights: '固定版本模型权重', detailsEmpty: '运行预测后可查看完整结果。',
    note: '使用公开的种子 42–46 检查点。浏览器结果与 Python 后端可能存在微小浮点差异。请求数据仅保存在本页内存中。',
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
    meta: 'Five-model ensemble · Public checkpoints · On-device CPU inference', subtitle: 'One cell · 336 hours of history → 24 hours of forecast',
    github: 'GitHub repository', website: 'Project website', languages: 'Interface language',
    reset: 'Reset to sample', choose: 'Choose a 336-hour CSV', drop: 'Drop a CSV here, or click to choose', builtin: 'Built-in synthetic sample', localFile: 'Local file', emptyFile: 'No CSV selected',
    privacy: 'Files stay in your browser and run on your CPU.', fileHint: 'UTF-8 · 336 rows · NIL or blanks for gaps · 5 MB max',
    sampleDownload: 'Download sample CSV', clear: 'Clear file', pattern: 'Missingness pattern', patterns: ['Complete history', 'Random whole-hour loss', 'Contiguous block loss', 'Recent-tail outage', 'Asynchronous indicator loss'], rate: 'Additional missingness',
    run: 'Run forecast', cancel: 'Cancel', ready: 'Choose a file or reset to the sample, then run the forecast.', loading: 'Loading and verifying public models…', running: n => `Computing on this device: ${n} / 5 models`, done: 'Forecast complete', preview: 'Automatic sample preview', cancelled: 'Cancelled. Run the forecast to start again.',
    chart: 'History and the next 24 hours', history: 'Observed history', forecast: 'Five-model ensemble', envelope: 'Audited envelope', missing: 'Missing position',
    fullHistory: '336-hour history', recentHistory: 'Last 72 hours', range: 'History display range', chartHelp: 'The shaded envelope shows the model’s checkable bounds. Red ticks mark missing positions.',
    scenario: 'Scenario', configured: 'Configured removal', actuallyRemoved: 'Actually removed', observed: 'Final observed', audit: 'Audit', maskLabel: 'Mask', boundLabel: 'Bound', passed: 'PASS', failed: 'FAIL',
    runtime: 'On-device inference', milliseconds: 'ms',
    routing: 'Ensemble routing summary', routingHelp: 'Mean expert values and routing weights across five members at the selected hour. This view summarizes internal routing; these two means do not exactly reconstruct the final ensemble prediction.',
    indicator: 'Indicator', horizon: 'Future hour', future: h => `Future hour ${h}`,
    metrics: ['UL active users', 'DL active users', 'Average used DL PRBs', 'Average used UL PRBs'],
    expertValue: 'Ensemble-mean expert value', meanWeight: 'Mean routing weight', shortValue: 'Value', shortWeight: 'Weight', unavailable: 'Unavailable expert',
    experts: ['Previous day', 'Previous week', 'Two weeks earlier', '7-day same-hour median', '14-day same-hour median', 'Limited weekly trend', 'Window median', 'Frozen training prior'],
    shortExperts: ['1 d', '7 d', '14 d', 'Med\n7', 'Med\n14', 'Trend', 'Local', 'Prior'],
    advanced: 'Advanced details', table: 'Full 24 × 4 forecast', expertTable: 'Eight experts at the selected hour', memberTable: 'Ensemble members',
    time: 'Time', tableMetrics: ['UL users', 'DL users', 'DL PRBs', 'UL PRBs'],
    expert: 'Seasonal expert', available: 'Available', yes: 'Yes', no: 'No', support: 'Support', seed: 'Seed', config: 'Configuration', memberPrediction: 'Selected-hour forecast', residual: 'Log-space correction', bound: 'Residual bound',
    csv: 'Download forecast CSV', auditDownload: 'Download audit JSON',
    downloadHint: 'Run the forecast to enable CSV and audit JSON downloads. The automatic preview is for inspection.', downloadReady: 'The forecast and per-member audit record are ready. Downloads are generated in your browser.',
    inputFormat: 'Input schema', modelWeights: 'Pinned model weights', detailsEmpty: 'Run a forecast to inspect the complete results.',
    note: 'Uses the public seed 42–46 checkpoints. Browser and Python results may differ slightly due to floating-point arithmetic. Request data stays in page memory.',
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
  const localURL = (path) => new URL(path, projectURL).href;
  const icon = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M8 13h8M8 17h6"/></svg>';
  root.innerHTML = `
    <header class="local-demo-header">
      <nav class="local-demo-languages" aria-label="${t.languages}"><a href="${localURL('local-demo/')}" ${!zh ? 'aria-current="page"' : ''}>English</a><a href="${localURL('zh/local-demo/')}" ${zh ? 'aria-current="page"' : ''}>中文</a></nav>
      <p class="local-demo-meta">${t.meta}</p><p class="local-demo-subtitle">${t.subtitle}</p>
      <nav class="local-demo-links"><a href="https://github.com/rudykon/WLCR-SEA_Predictor" target="_blank" rel="noopener noreferrer">${t.github}</a><a href="${localURL(zh ? 'zh/' : '')}" target="_blank" rel="noopener noreferrer">${t.website}</a></nav>
    </header>
    <div class="local-demo-workspace">
      <section class="local-demo-controls" aria-label="${t.choose}">
        <button type="button" id="local-reset" class="md-button local-demo-reset" disabled>${t.reset}</button>
        <label class="local-demo-upload-label" for="local-file">${t.choose}</label>
        <div class="local-demo-dropzone" id="local-dropzone">
          <input id="local-file" type="file" accept=".csv,text/csv" disabled aria-describedby="local-file-hint local-privacy">
          <span class="local-demo-file-icon">${icon}</span>
          <strong id="local-filename">synthetic_traffic.csv</strong><span id="local-file-kind">${t.builtin}</span><span class="local-demo-drop-hint">${t.drop}</span>
        </div>
        <div class="local-demo-file-actions"><button type="button" id="local-sample" disabled>${t.sampleDownload}</button><button type="button" id="local-clear" disabled>${t.clear}</button></div>
        <p id="local-file-hint" class="local-demo-help">${t.fileHint}</p><p id="local-privacy" class="local-demo-privacy">${t.privacy}</p>
        <div class="local-demo-missing-controls">
          <label for="local-pattern">${t.pattern}</label><select id="local-pattern">${scenarios.map((s, i) => `<option value="${s}">${t.patterns[i]}</option>`).join('')}</select>
          <label class="local-demo-slider-label" for="local-rate">${t.rate}<output id="local-rate-value" for="local-rate">0%</output></label>
          <input id="local-rate" type="range" min="0" max="80" step="5" value="0" disabled><div class="local-demo-range-ends"><span>0%</span><span>80%</span></div>
        </div>
        <button type="button" id="local-run" class="md-button md-button--primary" disabled>${t.run}</button><button type="button" id="local-cancel" class="md-button" hidden>${t.cancel}</button>
        <p id="local-error" role="alert" hidden></p>
      </section>
      <section class="local-demo-forecast" aria-label="${t.chart}">
        <div class="local-demo-forecast-card"><div class="local-demo-chart-toolbar"><h2 id="local-chart-title">${t.chart}</h2><label class="local-demo-sr" for="local-range">${t.range}</label><select id="local-range" aria-label="${t.range}"><option value="336">${t.fullHistory}</option><option value="72">${t.recentHistory}</option></select></div>
          <div class="local-demo-legend"><span class="local-legend-history">${t.history}</span><span class="local-legend-forecast">${t.forecast}</span><span class="local-legend-envelope">${t.envelope}</span><span class="local-legend-missing">${t.missing}</span></div>
          <div id="local-chart" aria-label="${t.chart}"></div><p class="local-demo-help">${t.chartHelp}</p>
        </div>
        <div class="local-demo-status-wrap"><table class="local-demo-status-table"><thead><tr>${[t.scenario, t.configured, t.actuallyRemoved, t.observed, t.audit].map(x => `<th>${x}</th>`).join('')}</tr></thead><tbody id="local-status-tbody"><tr><td colspan="5">${t.loading}</td></tr></tbody></table></div>
        <p class="local-demo-result-meta"><span id="local-window"></span><span id="local-compute"></span></p><p id="local-status" role="status" aria-live="polite">${t.loading}</p>
      </section>
    </div>
    <section class="local-demo-routing-section" aria-labelledby="local-routing-heading"><h2 id="local-routing-heading">${t.routing}</h2><p class="local-demo-help">${t.routingHelp}</p>
      <div class="local-demo-view-controls"><div><label for="local-metric">${t.indicator}</label><select id="local-metric">${t.metrics.map((s, i) => `<option value="${i}" ${i === 2 ? 'selected' : ''}>${s}</option>`).join('')}</select></div><div><label class="local-demo-slider-label" for="local-horizon">${t.horizon}<output id="local-horizon-value" for="local-horizon">1</output></label><input id="local-horizon" type="range" min="1" max="24" step="1" value="1"><div class="local-demo-range-ends"><span>1</span><span>24</span></div></div></div>
      <div class="local-demo-expert-card"><h3 id="local-routing-title">${t.detailsEmpty}</h3><div class="local-demo-legend"><span class="local-legend-history">${t.expertValue}</span><span class="local-legend-weight">${t.meanWeight}</span><span class="local-legend-unavailable">${t.unavailable}</span></div><div id="local-routing" aria-label="${t.routing}"></div></div>
    </section>
    <details id="local-advanced" class="local-demo-advanced"><summary>${t.advanced}</summary><div id="local-results" hidden>
      <h3>${t.table}</h3><div class="local-demo-table"><table><thead><tr><th>${t.time}</th>${t.tableMetrics.map(x => `<th>${x}</th>`).join('')}</tr></thead><tbody id="local-tbody"></tbody></table></div>
      <h3>${t.expertTable}</h3><div class="local-demo-table"><table><thead><tr>${[t.expert, t.available, t.support, t.expertValue, t.meanWeight].map(x => `<th>${x}</th>`).join('')}</tr></thead><tbody id="local-expert-tbody"></tbody></table></div>
      <h3>${t.memberTable}</h3><div class="local-demo-table"><table><thead><tr>${[t.seed, t.config, t.memberPrediction, t.residual, t.bound, 'Checkpoint SHA-256'].map(x => `<th>${x}</th>`).join('')}</tr></thead><tbody id="local-member-tbody"></tbody></table></div>
    </div><p id="local-download-hint" class="local-demo-help">${t.downloadHint}</p><div class="local-demo-downloads"><button type="button" id="local-csv" class="md-button" disabled>${t.csv}<small>wlcr-sea-forecast.csv</small></button><button type="button" id="local-audit" class="md-button" disabled>${t.auditDownload}<small>wlcr-sea-audit.json</small></button></div>
    <p class="local-demo-help"><a href="${localURL((zh ? 'zh/' : '') + 'guide/data-contract/')}">${t.inputFormat}</a> · <a href="https://huggingface.co/config-h/WLCR-SEA-Predictor/tree/eb4447f4ebab8f9caa003d92c838ed8e750963bd" target="_blank" rel="noopener noreferrer">${t.modelWeights}</a></p><p class="local-demo-help">${t.note}</p></details>`;
  const el = id => root.querySelector(`#local-${id}`);
  const number = new Intl.NumberFormat(zh ? 'zh-CN' : 'en', { maximumFractionDigits: 3 });
  const percent = x => `${(x * 100).toFixed(1)}%`;
  let core, view, sampleText, csvText, request, output, worker, busy = false, revision = 0, metric = 2, hour = 0, displayHours = 336, automatic = false, envelope;
  const showError = code => { el('error').textContent = t.errors[code] || t.errors.unknown; el('error').hidden = false; root.dataset.state = 'error'; };
  function setBusy(value) {
    busy = value; el('cancel').hidden = !value;
    for (const id of ['reset', 'pattern', 'file', 'run', 'clear']) el(id).disabled = value;
    el('reset').disabled = value || !sampleText;
    el('rate').disabled = value || el('pattern').value === 'none';
    el('run').disabled = value || !request;
    if (value) { el('error').hidden = true; root.dataset.state = 'running'; }
  }
  function invalidate() {
    output = undefined; envelope = undefined; el('results').hidden = true; el('csv').disabled = true; el('audit').disabled = true;
    el('compute').textContent = ''; el('window').textContent = ''; el('status').textContent = t.ready; el('error').hidden = true; root.dataset.state = 'idle';
    el('status-tbody').replaceChildren(); const row = el('status-tbody').insertRow(), cell = row.insertCell(); cell.colSpan = 5; cell.textContent = t.ready;
    el('download-hint').textContent = t.downloadHint; el('routing-title').textContent = t.detailsEmpty; el('routing').replaceChildren();
    el('tbody').replaceChildren(); el('expert-tbody').replaceChildren(); el('member-tbody').replaceChildren(); drawChart();
  }
  function download(name, contents, type) {
    const url = URL.createObjectURL(new Blob([contents], { type }));
    const a = document.createElement('a'); a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function svgCanvas(box, height, label) {
    const width = Math.max(280, box.clientWidth || 700), ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', label);
    const add = (tag, attrs, text, parent = svg) => { const e = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v)); if (text !== undefined) e.textContent = text; parent.append(e); return e; };
    return { width, svg, add };
  }
  function axisScale(values) {
    const range = Math.max(1, ...values) * 1.08;
    const magnitude = 10 ** Math.floor(Math.log10(range / 5));
    const target = range / 5 / magnitude;
    const factor = [1, 2, 2.5, 5, 10].filter(x => x <= target).at(-1) || 1;
    const step = magnitude * factor;
    return { max: Math.ceil(range / step) * step, step };
  }
  function drawChart() {
    const box = el('chart');
    if (!request) { box.replaceChildren(); return; }
    const history = output?.request.values || request.values, result = output?.result;
    const { width, svg, add } = svgCanvas(box, 304, `${t.metrics[metric]}: ${t.chart}`);
    const left = width < 450 ? 48 : 57, right = width - 14, top = 18, bottom = 258;
    const first = 336 - displayHours, last = 359, points = history.slice(first).map(row => row[metric]).filter(x => x !== null);
    if (result) { points.push(...result.prediction.map(row => row[metric])); points.push(...envelope.upper.map(row => row[metric])); }
    const { max, step: tickStep } = axisScale(points), x = i => left + (i - first) / (last - first) * (right - left), y = v => bottom - v / max * (bottom - top);
    el('chart-title').textContent = `${t.metrics[metric]} · ${displayHours === 336 ? t.fullHistory : t.recentHistory} → 24 h`;
    const divider = x(336);
    for (let v = 0; v <= max + tickStep * .01; v += tickStep) { add('line', { x1: left, y1: y(v), x2: right, y2: y(v), stroke: 'var(--wl-border)', 'stroke-width': .7 }); add('text', { x: left - 7, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--wl-muted)', 'font-size': 11 }, number.format(v)); }
    add('line', { x1: divider, x2: divider, y1: top, y2: bottom, stroke: 'var(--wl-muted)', 'stroke-dasharray': '3 4' });
    const path = (rows, offset, column) => { let d = '', open = false; for (let i = 0; i < rows.length; i++) { const v = rows[i][column]; if (v === null) { open = false; continue; } d += `${open ? 'L' : 'M'}${x(offset + i).toFixed(2)},${y(v).toFixed(2)} `; open = true; } return d; };
    if (result) {
      const band = envelope.upper.map((row, h) => `${h ? 'L' : 'M'}${x(336 + h)},${y(row[metric])}`).join(' ') + ' ' + [...envelope.lower].reverse().map((row, i) => `L${x(359 - i)},${y(row[metric])}`).join(' ') + ' Z';
      add('path', { d: band, class: 'local-demo-envelope', fill: 'var(--wl-teal)', opacity: .15 });
    }
    add('path', { d: path(history.slice(first), first, metric), fill: 'none', stroke: 'var(--wl-blue)', 'stroke-width': 1.5, 'stroke-linejoin': 'round' });
    for (let i = first; i < 336; i++) {
      const value = history[i][metric];
      if (value === null) add('line', { x1: x(i), x2: x(i), y1: bottom - 5, y2: bottom, stroke: 'var(--local-missing)', 'stroke-width': 1.1 });
      else if (i === first || i === 335 || history[i - 1][metric] === null || history[i + 1][metric] === null) add('circle', { cx: x(i), cy: y(value), r: 1.4, fill: 'var(--wl-blue)' });
    }
    if (result) {
      add('path', { d: path(result.prediction, 336, metric), fill: 'none', stroke: 'var(--local-forecast)', 'stroke-width': 2.1 });
      for (let h = 0; h < 24; h += 3) add('circle', { cx: x(336 + h), cy: y(result.prediction[h][metric]), r: 1.7, fill: 'var(--local-forecast)' });
      add('circle', { cx: x(336 + hour), cy: y(result.prediction[hour][metric]), r: 3.8, fill: 'var(--wl-surface)', stroke: 'var(--local-weight)', 'stroke-width': 2, class: 'local-demo-selected-hour' });
    }
    const tickCount = width < 450 ? 3 : 5;
    for (let n = 0; n < tickCount; n++) {
      const i = Math.round(first + n * (last - first) / (tickCount - 1));
      const time = i < 336 ? request.times[i] : request.times.at(-1) + (i - 335) * 3600000;
      const text = core.formatTimestamp(time).slice(5, displayHours === 336 ? 10 : 16);
      add('text', { x: x(i), y: bottom + 24, 'text-anchor': n === 0 ? 'start' : n === tickCount - 1 ? 'end' : 'middle', fill: 'var(--wl-muted)', 'font-size': 11 }, text);
    }
    box.replaceChildren(svg);
  }
  function fillRows(body, rows) {
    body.replaceChildren();
    for (const cells of rows) { const tr = document.createElement('tr'); for (const value of cells) { const td = document.createElement('td'); td.textContent = String(value); tr.append(td); } body.append(tr); }
  }
  function drawRouting() {
    if (!output) return;
    const result = output.result, rows = view.routingAt(result, metric, hour), box = el('routing');
    el('routing-title').textContent = `${t.metrics[metric]} · ${t.future(hour + 1)} · ${core.formatTimestamp(result.forecastTimes[hour])}`;
    const { width, svg, add } = svgCanvas(box, 340, `${t.routing}: ${t.metrics[metric]}, ${t.future(hour + 1)}`);
    const left = 50, right = width - 48, top = 30, bottom = 274, step = (right - left) / 8;
    const { max, step: tickStep } = axisScale(rows.map(r => r.value || 0)), y = v => bottom - v / max * (bottom - top), wy = v => bottom - v * (bottom - top);
    add('text', { x: left, y: 15, fill: 'var(--wl-muted)', 'font-size': 11 }, width < 500 ? t.shortValue : t.expertValue);
    add('text', { x: right, y: 15, fill: 'var(--local-weight)', 'text-anchor': 'end', 'font-size': 11 }, `${width < 500 ? t.shortWeight : t.meanWeight} (%)`);
    const defs = add('defs', {}), pattern = add('pattern', { id: 'local-unavailable-hatch', width: 5, height: 5, patternUnits: 'userSpaceOnUse' }, undefined, defs);
    add('rect', { width: 5, height: 5, fill: 'var(--wl-soft)' }, undefined, pattern); add('path', { d: 'M0 5L5 0', stroke: 'var(--wl-muted)', 'stroke-width': .7 }, undefined, pattern);
    for (let v = 0; v <= max + tickStep * .01; v += tickStep) {
      add('line', { x1: left, x2: right, y1: y(v), y2: y(v), stroke: 'var(--wl-border)', 'stroke-width': .7 });
      add('text', { x: left - 6, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--wl-muted)', 'font-size': 10 }, number.format(v));
    }
    for (let i = 0; i <= 4; i++) add('text', { x: right + 6, y: wy(i / 4) + 4, fill: 'var(--local-weight)', 'font-size': 10 }, String(i * 25));
    for (let j = 0; j < 8; j++) {
      const row = rows[j], center = left + (j + .5) * step, barWidth = step * .58, height = row.available ? bottom - y(row.value) : 5;
      const bar = add('rect', { x: center - barWidth / 2, y: bottom - height, width: barWidth, height, fill: row.available ? 'var(--wl-blue)' : 'url(#local-unavailable-hatch)', opacity: .88, 'data-expert': j });
      add('title', {}, `${t.experts[j]}: ${row.available ? number.format(row.value) : t.no}, ${percent(row.weight)}`, bar);
      const label = add('text', { x: center, y: bottom + 19, 'text-anchor': 'middle', fill: 'var(--wl-muted)', 'font-size': width < 500 ? 9 : 11 });
      t.shortExperts[j].split('\n').forEach((line, index) => add('tspan', { x: center, dy: index ? 12 : 0 }, line, label));
    }
    add('path', { d: rows.map((r, j) => `${j ? 'L' : 'M'}${left + (j + .5) * step},${wy(r.weight)}`).join(' '), fill: 'none', stroke: 'var(--local-weight)', 'stroke-width': 2 });
    rows.forEach((r, j) => add('circle', { cx: left + (j + .5) * step, cy: wy(r.weight), r: 3, fill: 'var(--local-weight)' }));
    box.replaceChildren(svg);
    fillRows(el('expert-tbody'), rows.map((r, j) => [t.experts[j], r.available ? t.yes : t.no, r.support.toFixed(4), r.value === null ? '—' : number.format(r.value), percent(r.weight)]));
    fillRows(el('member-tbody'), result.members.map(m => [m.seed, m.selected_config.name, number.format(m.prediction[hour][metric]), m.residual_log[hour][metric].toFixed(6), m.selected_config.residual_bound, m.checkpoint_sha256]));
    Array.from(el('tbody').rows).forEach((tr, h) => { tr.classList.toggle('local-demo-selected-row', h === hour); if (h === hour) tr.setAttribute('aria-selected', 'true'); else tr.removeAttribute('aria-selected'); });
  }
  function showResult(data) {
    output = data; envelope = view.ensembleEnvelope(data.result); root.dataset.state = 'done'; root.dataset.preview = automatic ? 'true' : 'false';
    el('status').textContent = automatic ? t.preview : t.done;
    el('compute').textContent = `${t.runtime}: ${number.format(data.elapsedMs)} ${t.milliseconds}`;
    el('window').textContent = `${request.cell} · ${core.formatTimestamp(data.result.forecastTimes[0])} – ${core.formatTimestamp(data.result.forecastTimes[23])}`;
    const r = data.request, maskOK = data.result.checks.max_unavailable_weight === 0, boundOK = data.result.checks.max_envelope_violation === 0;
    fillRows(el('status-tbody'), [[t.patterns[scenarios.indexOf(r.scenario)], percent(r.requestedRate), percent(r.removedFraction), percent(r.observed / 1344), `${t.maskLabel}${zh ? '' : ' '}${maskOK ? t.passed : t.failed} · ${t.boundLabel}${zh ? '' : ' '}${boundOK ? t.passed : t.failed}`]]);
    fillRows(el('tbody'), data.result.prediction.map((row, i) => [core.formatTimestamp(data.result.forecastTimes[i]), ...row.map(x => number.format(x))]));
    el('results').hidden = false; el('csv').disabled = automatic; el('audit').disabled = automatic;
    el('download-hint').textContent = automatic ? t.downloadHint : t.downloadReady; drawChart(); drawRouting();
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
  function runForecast(preview = false) {
    if (!request || busy) return;
    invalidate(); automatic = preview; setBusy(true); el('status').textContent = t.loading;
    try { getWorker().postMessage({ csv: csvText, scenario: el('pattern').value, rate: Number(el('rate').value) / 100 }); }
    catch { setBusy(false); showError('worker'); }
  }
  async function useSample(run = true, preview = false) {
    if (!core || !sampleText || busy) return;
    ++revision; el('file').value = ''; csvText = sampleText; request = core.parseRequest(csvText);
    el('filename').textContent = 'synthetic_traffic.csv'; el('file-kind').textContent = t.builtin; invalidate(); setBusy(false);
    if (run) runForecast(preview);
  }
  async function loadFile(file) {
    if (busy || !file) return;
    const version = ++revision; request = undefined; csvText = undefined; el('file').value = ''; invalidate(); el('run').disabled = true;
    el('filename').textContent = file.name; el('file-kind').textContent = `${t.localFile} · ${(file.size / 1024).toFixed(1)} KiB`;
    try {
      if (!file.name.toLowerCase().endsWith('.csv')) throw new Error('invalid_extension');
      if (file.size > 5 * 1024 * 1024) throw new Error('file_too_large');
      let text;
      try { text = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer()); } catch { throw new Error('invalid_encoding'); }
      if (version !== revision) return;
      const parsed = core.parseRequest(text); csvText = text; request = parsed; invalidate(); el('run').disabled = false;
    } catch (error) { if (version === revision) showError(error.message); }
  }
  el('file').addEventListener('change', () => loadFile(el('file').files[0]));
  const dropzone = el('dropzone');
  for (const name of ['dragenter', 'dragover']) dropzone.addEventListener(name, event => { event.preventDefault(); if (!busy) dropzone.classList.add('local-demo-dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('local-demo-dragover'));
  dropzone.addEventListener('drop', event => { event.preventDefault(); dropzone.classList.remove('local-demo-dragover'); if (!busy) loadFile(event.dataTransfer.files[0]); });
  el('reset').addEventListener('click', () => useSample(true));
  el('clear').addEventListener('click', () => { if (busy) return; ++revision; request = undefined; csvText = undefined; el('file').value = ''; el('filename').textContent = t.emptyFile; el('file-kind').textContent = t.drop; invalidate(); setBusy(false); });
  el('pattern').addEventListener('change', () => { invalidate(); if (el('pattern').value === 'none') el('rate').value = 0; else if (Number(el('rate').value) === 0) el('rate').value = 20; el('rate-value').textContent = `${el('rate').value}%`; el('rate').disabled = el('pattern').value === 'none'; });
  el('rate').addEventListener('input', () => { el('rate-value').textContent = `${el('rate').value}%`; invalidate(); });
  el('metric').addEventListener('change', () => { metric = Number(el('metric').value); drawChart(); drawRouting(); });
  el('horizon').addEventListener('input', () => { hour = Number(el('horizon').value) - 1; el('horizon-value').textContent = String(hour + 1); drawChart(); drawRouting(); });
  el('range').addEventListener('change', () => { displayHours = Number(el('range').value); drawChart(); });
  el('run').addEventListener('click', () => runForecast(false));
  el('cancel').addEventListener('click', () => { worker?.terminate(); worker = undefined; setBusy(false); invalidate(); el('status').textContent = t.cancelled; });
  el('sample').addEventListener('click', () => download('synthetic_traffic.csv', sampleText, 'text/csv;charset=utf-8'));
  el('csv').addEventListener('click', () => { if (output && !automatic) download('wlcr-sea-forecast.csv', core.forecastCSV(request, output.result), 'text/csv;charset=utf-8'); });
  el('audit').addEventListener('click', () => {
    if (!output || automatic) return;
    const audit = { schema: 'wlcr-sea-browser-audit/v1', runtime: 'wlcr-sea-browser/1', executed_on: 'visitor_device_cpu',
      input_text_sha256: output.inputTextSHA256, input_hash_encoding: 'UTF-8 decoded text with BOM removed',
      model: output.result.model, request: output.request, forecast_times: output.result.forecastTimes.map(core.formatTimestamp),
      prediction: output.result.prediction, members: output.result.members, checks: output.result.checks,
      inference_ms: output.elapsedMs,
      numerical_implementation: 'Public float32 tensors; JavaScript accumulation and erf GELU (not bitwise PyTorch identity).' };
    download('wlcr-sea-audit.json', JSON.stringify(audit, null, 2), 'application/json');
  });
  for (const id of ['chart', 'routing']) {
    let oldWidth = 0;
    new ResizeObserver(entries => { const width = Math.round(entries[0].contentRect.width); if (width !== oldWidth) { oldWidth = width; if (core) { if (id === 'chart') drawChart(); else drawRouting(); } } }).observe(el(id));
  }
  window.addEventListener('pagehide', () => worker?.terminate());
  (async () => {
    try {
      [core, view] = await Promise.all([import(new URL('local-demo-core.mjs', scriptURL)), import(new URL('local-demo-view.mjs', scriptURL))]);
      setBusy(false);
      const response = await fetch(new URL('../assets/local-demo/sample.csv', scriptURL));
      if (!response.ok) throw new Error('sample');
      sampleText = await response.text(); el('sample').disabled = false; el('reset').disabled = false;
      if (!request && revision === 0) await useSample(true, true);
    } catch (error) { showError(error.message === 'sample' ? 'sample' : 'worker'); }
  })();
})();
