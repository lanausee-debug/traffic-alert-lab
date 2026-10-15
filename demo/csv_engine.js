/* Browser/Node implementation of traffic-alert-csv-v1. No network or storage. */
(function (root) {
  'use strict';
  const MAX_ROWS = 20000;
  const iso = value => new Date(value).toISOString().replace('.000Z', '+00:00');
  function timestamp(text) {
    const match = text.match(/^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(Z|[+-]\d{2}:\d{2})$/);
    if (!match) throw Error('时间需带秒和时区，例如 2020-01-01T08:00:00+08:00');
    const [, year, month, day, hour, minute, second, offset] = match;
    const parts = [year, month, day, hour, minute, second].map(Number);
    const date = new Date(0);
    date.setUTCFullYear(parts[0], parts[1] - 1, parts[2]);
    date.setUTCHours(parts[3], parts[4], parts[5], 0);
    const actual = [date.getUTCFullYear(), date.getUTCMonth() + 1, date.getUTCDate(), date.getUTCHours(), date.getUTCMinutes(), date.getUTCSeconds()];
    if (parts[0] < 1 || parts.some((value, i) => value !== actual[i])) throw Error('无效日期或时间');
    let minutes = 0;
    if (offset !== 'Z') {
      const hours = Number(offset.slice(1, 3)), mins = Number(offset.slice(4));
      if (hours > 23 || mins > 59) throw Error('无效时区偏移');
      minutes = (hours * 60 + mins) * (offset[0] === '-' ? -1 : 1);
    }
    const result = date.getTime() - minutes * 60000;
    if (new Date(result).getUTCFullYear() < 1 || new Date(result).getUTCFullYear() > 9999) throw Error('时间超出范围');
    return result;
  }
  function parseCSV(text) {
    if (typeof text !== 'string' || text.length > 2 * 1024 * 1024) throw Error('网页版限制 2 MiB 文本、20,000 行');
    text = text.replace(/^\uFEFF/, '');
    const table = [];
    let row = [], field = '', quoted = false, closed = false;
    function endField() { row.push(field); field = ''; closed = false; }
    function endRow() {
      endField();
      if (!(row.length === 1 && row[0] === '')) table.push(row);
      row = [];
      if (table.length > MAX_ROWS + 1) throw Error('网页版最多 20,000 行数据');
    }
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (quoted) {
        if (ch === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; }
          else { quoted = false; closed = true; }
        } else field += ch;
      } else if (ch === ',') endField();
      else if (ch === '\r' || ch === '\n') { if (ch === '\r' && text[i + 1] === '\n') i++; endRow(); }
      else if (ch === '"' && field === '' && !closed) quoted = true;
      else if (closed || ch === '"') throw Error('CSV 引号格式错误');
      else field += ch;
    }
    if (quoted) throw Error('CSV 引号没有闭合');
    if (field || row.length || closed) endRow();
    if (table.length < 2) throw Error('CSV 没有数据');
    const header = table.shift();
    if (new Set(header).size !== header.length) throw Error('CSV 列名不能重复');
    const required = ['station_id', 'timestamp', 'score', 'is_event'];
    const indexes = required.map(name => header.indexOf(name));
    if (indexes.some(index => index < 0)) throw Error('CSV 需要 station_id,timestamp,score,is_event 四列');
    return table.map((cells, index) => {
      try {
        if (cells.length !== header.length) throw Error('列数与表头不一致');
        const [station_id, time, scoreText, label] = indexes.map(i => cells[i].trim());
        if (!station_id) throw Error('站点编号不能为空');
        if (!['', '0', '1'].includes(label)) throw Error('is_event 只能为 0、1 或空白');
        if (scoreText && !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(scoreText)) throw Error('score 必须为数值或空白');
        const score = scoreText === '' ? null : Number(scoreText);
        if (score !== null && (!Number.isFinite(score) || score < 0 || score > 1)) throw Error('score 必须在 0 到 1 之间');
        return {station_id, time: timestamp(time), score, is_event: label === '' ? null : Number(label)};
      } catch (error) { throw Error(`数据记录 ${index + 1}：${error.message}`); }
    });
  }
  function compareText(a, b) {
    const left = Array.from(a, ch => ch.codePointAt(0)), right = Array.from(b, ch => ch.codePointAt(0));
    for (let i = 0; i < Math.min(left.length, right.length); i++) if (left[i] !== right[i]) return left[i] - right[i];
    return left.length - right.length;
  }
  function evaluate(rows, settings = {}) {
    const {threshold = 0.5, step_minutes = 5, horizon_minutes = 15, early_minutes = 10} = settings;
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw Error('阈值需在 0 到 1 之间');
    if (!Number.isInteger(step_minutes) || step_minutes <= 0 || !Number.isInteger(horizon_minutes) || horizon_minutes < 0 || horizon_minutes % step_minutes || !Number.isInteger(early_minutes) || early_minutes < 0 || early_minutes > horizon_minutes) throw Error('步长需为正整数；关联窗口需为步长的倍数；提前要求不能超过窗口');
    if (!rows.length || rows.length > MAX_ROWS) throw Error('数据行数应为 1 到 20,000');
    const ordered = [...rows].sort((a, b) => compareText(a.station_id, b.station_id) || a.time - b.time);
    const segments = [];
    let segment = [], previous = null, gaps = 0, unknown = 0, missing = 0;
    for (const row of ordered) {
      const same = previous !== null && row.station_id === previous.station_id;
      const delta = same ? row.time - previous.time : null;
      if (same && (delta <= 0 || delta % (step_minutes * 60000))) throw Error('时间重复或不符合该站点的采样网格');
      const gap = same && delta !== step_minutes * 60000;
      gaps += Number(gap);
      if (!same || gap || row.is_event === null) { if (segment.length) segments.push(segment); segment = []; }
      unknown += Number(row.is_event === null);
      missing += Number(row.score === null);
      if (row.is_event !== null) segment.push(row);
      previous = row;
    }
    if (segment.length) segments.push(segment);
    const horizon = horizon_minutes / step_minutes, events = [], alerts = [], excluded = [];
    let ticks = 0, falseAlarms = 0, duplicates = 0, excludedAlerts = 0;
    for (const part of segments) {
      const n = part.length, runs = [];
      for (let i = 0; i < n; i++) {
        if (part[i].is_event === 0) continue;
        const start = i;
        while (i + 1 < n && part[i + 1].is_event === 1) i++;
        const end = i, eligible = start > 0 && end < n - 1 && start >= horizon && end + horizon < n;
        const run = {start, end, eligible, event: null};
        if (eligible) {
          run.event = {station_id: part[0].station_id, onset: iso(part[start].time), end: iso(part[end].time), first_alert: null, lead_minutes: null};
          events.push(run.event);
        } else excluded.push({station_id: part[0].station_id, onset: iso(part[start].time), end: iso(part[end].time), reason: 'boundary event or insufficient known horizon context'});
        runs.push(run);
      }
      const eligible = part.map((_, i) => i + horizon < n && !runs.some(run => !run.eligible && run.start - horizon <= i && i <= run.end));
      ticks += eligible.filter(Boolean).length;
      let active = null;
      part.forEach((row, i) => {
        if (row.score === null) { active = null; return; }
        if (row.score < threshold) { active = false; return; }
        if (active === true) return;
        const leftCensored = active === null;
        active = true;
        if (leftCensored || !eligible[i]) {
          excludedAlerts++;
          alerts.push({station_id: row.station_id, timestamp: iso(row.time), status: 'excluded', reason: 'unknown episode start or ineligible context'});
          return;
        }
        const candidate = runs.find(run => run.eligible && run.start - horizon <= i && i <= run.end);
        let status, matched = null;
        if (!candidate) { falseAlarms++; status = 'false_alarm'; }
        else {
          matched = candidate.event;
          if (matched.first_alert === null) {
            matched.first_alert = iso(row.time);
            matched.lead_minutes = (part[candidate.start].time - row.time) / 60000;
            status = 'matched';
          } else { duplicates++; status = 'duplicate'; }
        }
        alerts.push({station_id: row.station_id, timestamp: iso(row.time), status, event_onset: matched ? matched.onset : null});
      });
    }
    const detected = events.filter(e => e.first_alert !== null).length;
    const early = events.filter(e => e.lead_minutes !== null && e.lead_minutes >= early_minutes).length;
    const exposure = ticks * step_minutes / 1440;
    return {protocol: 'traffic-alert-csv-v1', synthetic: false,
      settings: {threshold, step_minutes, horizon_minutes, early_minutes},
      metrics: {eligible_events: events.length, detected_events: detected, early_events: early,
        event_detection_rate: events.length ? detected / events.length : null, early_event_recall: events.length ? early / events.length : null,
        false_alarm_episodes: falseAlarms, duplicate_alert_episodes: duplicates, eligible_station_days: exposure,
        false_alarms_per_eligible_station_day: exposure ? falseAlarms / exposure : null},
      audit: {rows: rows.length, stations: new Set(rows.map(row => row.station_id)).size, known_segments: segments.length,
        sampling_gaps: gaps, unknown_label_rows: unknown, missing_score_rows: missing, excluded_event_runs: excluded.length,
        excluded_alert_episodes: excludedAlerts, eligible_decision_ticks: ticks},
      events, alerts, excluded_event_runs: excluded,
      limitations: ['Evaluation-only labels must never be model inputs.',
        'Use a threshold frozen on separate calibration data; this tool does not select it.',
        'A sustained positive episode can match only its earliest compatible event.',
        'Generic timestamp lead has no thesis-specific +one-step offset.',
        'Rates pool event counts and eligible exposure across stations; zero denominators return null.']};
  }
  root.TrafficCSV = {parseCSV, evaluate, timestamp};
})(globalThis);
