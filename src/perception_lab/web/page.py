# ruff: noqa: E501 (embedded HTML, CSS and JavaScript)
"""The single dashboard page (no external assets, no build step)."""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pet Monitor</title>
<style>
  :root { --bg: #f6f7f9; --fg: #1d2330; --muted: #677085; --card: #fff; --line: #dde1e8;
          --accent: #2a6fdb; --warn: #b54708; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #14171c; --fg: #e6e9ef; --muted: #9aa3b5; --card: #1d2128; --line: #2e333d;
            --accent: #6aa0ff; --warn: #f5a25d; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 system-ui, sans-serif; background: var(--bg); color: var(--fg); }
  header { padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex; gap: 16px;
           align-items: baseline; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr); gap: 16px;
         padding: 16px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  section { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
            padding: 12px; min-width: 0; }
  h2 { font-size: 15px; margin: 0 0 8px; }
  .muted { color: var(--muted); }
  #live { width: 100%; border-radius: 6px; background: #000; cursor: crosshair; display: block; }
  pre { background: var(--bg); padding: 8px; border-radius: 6px; overflow-x: auto; font-size: 12px; }
  .event { border-top: 1px solid var(--line); padding: 10px 0; }
  .event:first-child { border-top: 0; }
  .tag { display: inline-block; padding: 0 6px; border-radius: 4px; border: 1px solid var(--line);
         font-size: 12px; margin-right: 4px; }
  .thumbs img { height: 72px; margin: 4px 4px 0 0; border-radius: 4px; }
  .unc { color: var(--warn); font-size: 12px; }
  button, input { font: inherit; padding: 4px 8px; border-radius: 6px; border: 1px solid var(--line);
                  background: var(--card); color: var(--fg); }
  button:hover { border-color: var(--accent); }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
</style>
</head>
<body>
<header>
  <h1>Pet Monitor</h1>
  <span id="status" class="muted">connecting…</span>
</header>
<main>
  <div>
    <section>
      <h2>Live view</h2>
      <img id="live" hidden alt="Annotated live view">
      <p class="muted" id="nolive">The live view appears while <code>lab run --serve</code> is monitoring.</p>
      <p class="muted">Click the image to collect zone points; paste the snippet into your
      scene TOML.</p>
      <div class="row"><button id="clear">Clear points</button></div>
      <pre id="zone"># no points yet</pre>
    </section>
    <section style="margin-top:16px">
      <h2>Pipeline</h2>
      <pre id="pipeline">–</pre>
    </section>
  </div>
  <section>
    <div class="row" style="justify-content:space-between">
      <h2>Events</h2>
      <form id="search" class="row"><input id="q" placeholder="search descriptions">
        <button>Search</button></form>
    </div>
    <div id="events" class="muted">loading…</div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag);
  if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
let points = [];

async function getJSON(url) { const r = await fetch(url); return r.ok ? r.json() : null; }

async function refreshStatus() {
  const s = await getJSON('/api/status');
  if (!s) { $('status').textContent = 'dashboard unreachable'; return; }
  const p = s.pipeline;
  $('status').textContent = `${s.events_total} events` +
    (p ? ` · ${p.frames} frames · ${p.processing_fps} fps · ${p.active_tracks.length} tracks`
       : ' · not monitoring');
  $('pipeline').textContent = p ? JSON.stringify(
    {stage_ms: p.stage_ms, vlm: p.vlm, active_tracks: p.active_tracks,
     last_frame_age_s: p.last_frame_age_s}, null, 1) : 'not running (history only)';
  $('live').hidden = !s.live; $('nolive').hidden = s.live;
  if (s.live) $('live').src = '/api/live.jpg?t=' + Date.now();
}

function renderEvent(e) {
  const box = el('div', 'event');
  const head = el('div', 'row');
  head.append(el('strong', '', e.hypothesis.replaceAll('_', ' ')));
  head.append(el('span', 'tag', e.rule));
  (e.animal_classes || []).forEach((c, i) => head.append(el('span', 'tag', `${c} #${e.track_ids[i]}`)));
  if (e.zone) head.append(el('span', 'tag', e.zone));
  head.append(el('span', 'muted', `${e.detected_at} · ${(e.timestamp_end_s - e.timestamp_start_s).toFixed(0)} s`));
  box.append(head);
  const facts = Object.entries(e.observed_facts || {}).map(([k, v]) => `${k}: ${v}`).join(' · ');
  box.append(el('div', 'muted', 'Measured: ' + (facts || 'none')));
  const it = e.interpretation;
  if (e.interpretation_status === 'done' && it) {
    box.append(el('div', '', `VLM (${it.model}, ${it.certainty}): ${it.pet ? it.pet + ' · ' : ''}${it.activity} — ${it.description}`));
    (it.visual_observations || []).forEach((o) => box.append(el('div', 'muted', '• ' + o)));
  } else {
    box.append(el('div', 'muted', 'Interpretation: ' + e.interpretation_status));
  }
  (e.uncertainty || []).forEach((u) => box.append(el('div', 'unc', '⚠ ' + u)));
  const thumbs = el('div', 'thumbs');
  (e.evidence || []).forEach((ev) => { const img = el('img'); img.loading = 'lazy';
    img.src = '/media/' + ev.path; img.alt = ev.kind; thumbs.append(img); });
  box.append(thumbs);
  const review = el('div', 'row');
  review.append(el('span', 'muted', 'Review: ' + e.review_status + (e.review_label ? ` (${e.review_label})` : '')));
  for (const status of ['confirmed', 'rejected']) {
    const b = el('button', '', status === 'confirmed' ? 'Correct' : 'Wrong');
    b.onclick = async () => {
      const label = status === 'rejected' ? (prompt('What was it actually? (optional)') || null) : null;
      await fetch(`/api/events/${e.event_id}/review`, {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({status, label})});
      loadEvents();
    };
    review.append(b);
  }
  box.append(review);
  return box;
}

async function loadEvents(q) {
  const list = await getJSON(q ? '/api/search?q=' + encodeURIComponent(q) : '/api/events?limit=50');
  const root = $('events'); root.textContent = ''; root.className = '';
  if (!list || !list.length) { root.className = 'muted'; root.textContent = 'No events.'; return; }
  list.forEach((e) => root.append(renderEvent(e)));
}

$('live').addEventListener('click', (ev) => {
  const img = ev.target, r = img.getBoundingClientRect();
  const x = Math.round((ev.clientX - r.left) * img.naturalWidth / r.width);
  const y = Math.round((ev.clientY - r.top) * img.naturalHeight / r.height);
  points.push([x, y]);
  $('zone').textContent = '[[zones]]\\nname = "my_zone"\\ntags = ["restricted"]\\npolygon_xy_px = ' +
    JSON.stringify(points);
});
$('clear').onclick = () => { points = []; $('zone').textContent = '# no points yet'; };
$('search').onsubmit = (ev) => { ev.preventDefault(); loadEvents($('q').value.trim()); };
refreshStatus(); loadEvents();
setInterval(refreshStatus, 1000); setInterval(() => { if (!$('q').value) loadEvents(); }, 5000);
</script>
</body>
</html>
"""
