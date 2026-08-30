/* GODALGO terminal front end.
 *
 * The cluster is a canvas rather than DOM nodes: a few hundred neurons with
 * per-frame physics would thrash layout as elements, and hit-testing a circle
 * is cheaper than maintaining a DOM tree that changes every frame.
 *
 * Physics is deliberately gentle. Positions are data, not a game -- the motion
 * exists to make the cluster feel alive and to keep overlapping neurons
 * separable, not to be dramatic. Anything faster makes the display hard to read
 * and hard to click.
 */

const COLOR = {
  open:   [255, 140,  26],
  profit: [ 34, 224, 138],
  loss:   [255,  59,  71],
};

const state = {
  neurons: new Map(),   // id -> render node (position, velocity, data)
  selected: null,
  equityHistory: [],
  tab: 'trades',
  journal: { entries: [], summaries: [] },
  connected: false,
};

/* ---------------------------------------------------------------- utils */

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) =>
  n === null || n === undefined || !isFinite(n)
    ? '—'
    : n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (n, d = 2) => (n === null || !isFinite(n) ? '—' : (n * 100).toFixed(d) + '%');
const signed = (n, d = 2) => (n >= 0 ? '+' : '') + fmt(n, d);

function setSigned(el, value, formatter = signed) {
  el.textContent = formatter(value);
  el.classList.toggle('pos', value > 0);
  el.classList.toggle('neg', value < 0);
}

function shortTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toISOString().slice(5, 16).replace('T', ' ');
}

/* ------------------------------------------------------------- cluster */

const canvas = $('cluster-canvas');
const ctx = canvas.getContext('2d');
let W = 0, H = 0;

function resize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  W = rect.width; H = rect.height;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resize);

// A window resize listener alone is not enough: the canvas is a flex child, so
// it also changes size when the grid reflows or a sibling panel grows, with no
// window event. Stale dimensions let the soft walls sit outside the visible
// area and neurons get clipped at the edge.
if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);

/** Radius from notional, on a square root so a 100x larger position is 10x
 *  wider rather than 100x -- otherwise one big trade fills the panel. */
function radiusFor(n) {
  const base = Math.sqrt(Math.max(n.notional, 1)) * 0.42;
  return Math.max(5, Math.min(26, base));
}

function syncNeurons(list) {
  const seen = new Set();
  for (const data of list) {
    seen.add(data.id);
    let node = state.neurons.get(data.id);
    if (!node) {
      // Enter near the centre with a small outward drift, so new positions
      // visibly emerge from the middle rather than popping in at an edge.
      const angle = Math.random() * Math.PI * 2;
      // Seeded on a ring at a random radius so a batch of positions arriving at
      // once disperses immediately instead of resolving a pile-up at the centre.
      const ring = Math.min(W, H) * (0.10 + Math.random() * 0.32);
      node = {
        x: W / 2 + Math.cos(angle) * ring,
        y: H / 2 + Math.sin(angle) * ring,
        vx: Math.cos(angle) * 0.3,
        vy: Math.sin(angle) * 0.3,
        born: performance.now(),
        data,
      };
      state.neurons.set(data.id, node);
    } else {
      node.data = data;
    }
    node.r = radiusFor(data);
  }
  for (const id of [...state.neurons.keys()]) {
    if (!seen.has(id)) state.neurons.delete(id);
  }
  $('c-count').textContent = `${list.length} position${list.length === 1 ? '' : 's'}`;
  $('c-empty').style.display = list.length ? 'none' : 'flex';
}

function step() {
  const nodes = [...state.neurons.values()];
  if (!nodes.length) return;
  const cx = W / 2, cy = H / 2;

  // Containment radius sized to the panel, not to the node count. Pulling
  // everything to a single point packs the cluster into a ball and leaves the
  // rest of the panel empty; instead nodes are only pulled back once they
  // stray outside this radius, so they spread to fill the space available.
  const spread = Math.min(W, H) * 0.44;

  for (const a of nodes) {
    const dx = a.x - cx, dy = a.y - cy;
    const dist = Math.hypot(dx, dy) || 0.01;
    if (dist > spread) {
      const pull = (dist - spread) * 0.0022;
      a.vx -= (dx / dist) * pull;
      a.vy -= (dy / dist) * pull;
    }
    // Brownian jitter: the "alive" part.
    a.vx += (Math.random() - 0.5) * 0.05;
    a.vy += (Math.random() - 0.5) * 0.05;
  }

  // Repulsion at a range scaled to how much room each node can claim, so a
  // sparse cluster spreads out and a crowded one packs without overlapping.
  const room = Math.sqrt((W * H) / nodes.length);
  const reach = Math.max(38, Math.min(150, room * 0.78));

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      let dx = b.x - a.x, dy = b.y - a.y;
      const d2 = dx * dx + dy * dy;
      const min = a.r + b.r + reach;
      if (d2 < min * min) {
        const d = Math.sqrt(d2) || 0.01;
        const push = ((min - d) / min) * 0.16;
        dx /= d; dy /= d;
        a.vx -= dx * push; a.vy -= dy * push;
        b.vx += dx * push; b.vy += dy * push;
      }
    }
  }

  for (const a of nodes) {
    a.vx *= 0.955; a.vy *= 0.955;          // damping keeps motion calm
    a.x += a.vx; a.y += a.vy;
    // Soft walls: bounce inside the panel rather than drifting out of view.
    // Padding covers the label drawn beneath each neuron, not just its radius.
    const pad = a.r + 16;
    if (a.x < pad)      { a.x = pad;     a.vx = Math.abs(a.vx) * 0.6; }
    if (a.x > W - pad)  { a.x = W - pad; a.vx = -Math.abs(a.vx) * 0.6; }
    if (a.y < pad)      { a.y = pad;     a.vy = Math.abs(a.vy) * 0.6; }
    if (a.y > H - pad)  { a.y = H - pad; a.vy = -Math.abs(a.vy) * 0.6; }
  }
}

function draw(now) {
  ctx.clearRect(0, 0, W, H);

  // Backdrop grid — cheap depth, drawn under everything.
  ctx.strokeStyle = 'rgba(16,24,39,0.55)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < W; x += 44) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
  for (let y = 0; y < H; y += 44) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
  ctx.stroke();

  const nodes = [...state.neurons.values()];

  // Synapses between nearby neurons.
  ctx.lineWidth = 1;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.hypot(dx, dy);
      if (d > 190) continue;
      const alpha = (1 - d / 190) * 0.3;
      // Same-symbol positions are related, so their synapse reads amber.
      const same = a.data.symbol === b.data.symbol;
      ctx.strokeStyle = same
        ? `rgba(255,140,26,${alpha * 0.9})`
        : `rgba(53,214,240,${alpha * 0.4})`;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }

  for (const node of nodes) {
    const [r, g, bl] = COLOR[node.data.state] || COLOR.open;
    const isSel = state.selected === node.data.id;
    // Open positions breathe; closed ones are settled and hold steady.
    const pulse = node.data.is_open ? 0.72 + 0.28 * Math.sin(now / 420 + node.born) : 1;

    // Tight glow. A wide one looks impressive alone and turns a busy cluster
    // into an indistinct haze where nothing can be told apart or clicked.
    const halo = node.r * 1.9;
    const glow = ctx.createRadialGradient(node.x, node.y, node.r * 0.6, node.x, node.y, halo);
    glow.addColorStop(0, `rgba(${r},${g},${bl},${0.34 * pulse})`);
    glow.addColorStop(1, `rgba(${r},${g},${bl},0)`);
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(node.x, node.y, halo, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = `rgba(${r},${g},${bl},${0.82 * pulse})`;
    ctx.beginPath(); ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2); ctx.fill();

    ctx.strokeStyle = isSel ? '#ffffff' : `rgba(${r},${g},${bl},0.95)`;
    ctx.lineWidth = isSel ? 2 : 1;
    ctx.beginPath(); ctx.arc(node.x, node.y, node.r + (isSel ? 5 : 0), 0, Math.PI * 2); ctx.stroke();

    // Label only larger neurons, so the view stays readable when busy, and
    // never below the canvas edge where it would be clipped mid-glyph.
    if (node.r > 11) {
      const ly = node.y + node.r + 11;
      if (ly < H - 2) {
        ctx.fillStyle = 'rgba(234,242,255,0.82)';
        ctx.font = '9px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(node.data.symbol.split('/')[0], node.x, ly);
      }
    }
  }
}

function frame(now) {
  step();
  draw(now);
  requestAnimationFrame(frame);
}

canvas.addEventListener('click', (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  let hit = null, best = Infinity;
  for (const node of state.neurons.values()) {
    const d = Math.hypot(node.x - x, node.y - y);
    // Generous target: these are small, moving circles.
    if (d < node.r + 8 && d < best) { best = d; hit = node; }
  }
  state.selected = hit ? hit.data.id : null;
  renderDetail(hit ? hit.data : null);
});

/* ------------------------------------------------------------- panels */

function renderDetail(d) {
  const body = $('d-body');
  $('d-id').textContent = d ? d.id : 'overview';
  // With nothing selected the panel shows a cluster overview rather than an
  // empty box -- the space exists either way, so it may as well say something.
  if (!d) { body.innerHTML = clusterOverview(); return; }

  const pnlClass = d.net_pnl > 0 ? 'pos' : d.net_pnl < 0 ? 'neg' : '';
  body.innerHTML = `
    <dl class="kv">
      <dt>instrument</dt><dd class="amber">${d.symbol}</dd>
      <dt>side</dt><dd>${d.side.toUpperCase()}</dd>
      <dt>state</dt><dd class="${pnlClass}">${d.state.toUpperCase()}</dd>
      <dt>quantity</dt><dd>${fmt(d.quantity, 8)}</dd>
      <dt>notional</dt><dd>${fmt(d.notional)}</dd>
      <dt>open</dt><dd>${fmt(d.entry_price)}</dd>
      <dt>close</dt><dd>${d.exit_price === null ? '— open —' : fmt(d.exit_price)}</dd>
      <dt>mark</dt><dd>${d.mark_price === null ? '—' : fmt(d.mark_price)}</dd>
      <dt>opened</dt><dd>${shortTime(d.opened_at)}</dd>
      <dt>closed</dt><dd>${d.closed_at ? shortTime(d.closed_at) : '— open —'}</dd>
      <dt>held</dt><dd>${(d.duration_seconds / 60).toFixed(1)}m</dd>
      <dt>realised</dt><dd class="${d.realised_pnl > 0 ? 'pos' : d.realised_pnl < 0 ? 'neg' : ''}">${signed(d.realised_pnl)}</dd>
      <dt>unrealised</dt><dd class="${d.unrealised_pnl > 0 ? 'pos' : d.unrealised_pnl < 0 ? 'neg' : ''}">${signed(d.unrealised_pnl)}</dd>
      <dt>fees</dt><dd>${fmt(d.fees, 4)}</dd>
      <dt>net p&amp;l</dt><dd class="${pnlClass}">${signed(d.net_pnl)}</dd>
      <dt>return</dt><dd class="${pnlClass}">${pct(d.return_pct)}</dd>
      <dt>strategy</dt><dd>${d.strategy}</dd>
      <dt>regime</dt><dd>${d.regime}</dd>
      <dt>conviction</dt><dd>${fmt(d.conviction, 3)}</dd>
    </dl>`;
}

function clusterOverview() {
  const nodes = [...state.neurons.values()].map((n) => n.data);
  if (!nodes.length) return '<div class="empty">NO POSITIONS YET</div>';

  const closed = nodes.filter((n) => !n.is_open);
  const open = nodes.filter((n) => n.is_open);
  const best = closed.reduce((a, b) => (!a || b.net_pnl > a.net_pnl ? b : a), null);
  const worst = closed.reduce((a, b) => (!a || b.net_pnl < a.net_pnl ? b : a), null);

  const bySymbol = {};
  for (const n of nodes) {
    const s = (bySymbol[n.symbol] ||= { n: 0, pnl: 0 });
    s.n += 1; s.pnl += n.net_pnl;
  }
  const rows = Object.entries(bySymbol)
    .sort((a, b) => b[1].pnl - a[1].pnl)
    .map(([sym, v]) => `
      <div class="row" style="grid-template-columns:1fr 44px 80px;cursor:default">
        <span class="s">${sym}</span>
        <span class="t">${v.n}</span>
        <span class="p ${v.pnl >= 0 ? 'pos' : 'neg'}">${signed(v.pnl)}</span>
      </div>`).join('');

  return `
    <dl class="kv">
      <dt>tracked</dt><dd>${nodes.length}</dd>
      <dt>open</dt><dd class="amber">${open.length}</dd>
      <dt>closed</dt><dd>${closed.length}</dd>
      <dt>best</dt><dd class="pos">${best ? signed(best.net_pnl) + ' · ' + best.symbol : '—'}</dd>
      <dt>worst</dt><dd class="neg">${worst ? signed(worst.net_pnl) + ' · ' + worst.symbol : '—'}</dd>
    </dl>
    <div class="panel-title" style="border-top:1px solid var(--line)">
      <span>By instrument</span></div>
    ${rows}
    <p class="note" style="padding:10px 12px">Click any neuron for its full record.</p>`;
}

function renderHealth(h) {
  const p = Math.round(h.score * 100);
  $('l-fill').style.width = p + '%';
  $('l-head').style.left = p + '%';
  $('l-pct').textContent = p + '%';
  const label = $('l-label');
  label.textContent = h.label;
  label.className = 'label' + (h.label === 'NOMINAL' ? ' nominal' : h.label === 'DEGRADED' ? ' degraded' : '');
  $('l-detail').textContent = h.halted
    ? `HALTED — ${h.halt_reason || 'unknown'}`
    : `data ${h.data_age_seconds > 1e6 ? '—' : h.data_age_seconds.toFixed(0) + 's'} · ` +
      `reconnects ${h.reconnects} · errors ${h.errors}`;
}

function renderPnl(p) {
  setSigned($('p-net'), p.total_pnl);
  setSigned($('p-real'), p.realised);
  setSigned($('p-unreal'), p.unrealised);
  setSigned($('p-ret'), p.total_return, (v) => (v >= 0 ? '+' : '') + pct(v));
  $('p-win').textContent = pct(p.win_rate, 0);
  $('p-pf').textContent = p.profit_factor === null ? '∞' : fmt(p.profit_factor, 2);
  $('p-count').textContent = `${p.open_count} open · ${p.closed_count} closed`;
  setSigned($('h-pnl'), p.total_pnl);
  $('h-equity').textContent = fmt(p.equity);
}

const spark = $('equity-spark');
const sctx = spark.getContext('2d');

function drawSpark() {
  const rect = spark.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  spark.width = Math.round(rect.width * dpr);
  spark.height = Math.round(rect.height * dpr);
  sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  sctx.clearRect(0, 0, w, h);

  const pts = state.equityHistory;
  if (pts.length < 2) return;

  const lo = Math.min(...pts), hi = Math.max(...pts);
  const span = hi - lo || 1;
  const x = (i) => (i / (pts.length - 1)) * w;
  const y = (v) => h - 6 - ((v - lo) / span) * (h - 14);

  const up = pts[pts.length - 1] >= pts[0];
  const stroke = up ? '#22e08a' : '#ff3b47';

  sctx.beginPath();
  sctx.moveTo(x(0), y(pts[0]));
  pts.forEach((v, i) => sctx.lineTo(x(i), y(v)));
  sctx.lineTo(x(pts.length - 1), h); sctx.lineTo(x(0), h); sctx.closePath();
  const grad = sctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, up ? 'rgba(34,224,138,0.24)' : 'rgba(255,59,71,0.24)');
  grad.addColorStop(1, 'rgba(0,0,0,0)');
  sctx.fillStyle = grad; sctx.fill();

  sctx.beginPath();
  sctx.moveTo(x(0), y(pts[0]));
  pts.forEach((v, i) => sctx.lineTo(x(i), y(v)));
  sctx.strokeStyle = stroke; sctx.lineWidth = 1.4; sctx.stroke();
}

function renderBrain(b, h) {
  $('h-symbol').textContent = b.symbol;
  $('h-regime').textContent = b.regime;
  $('h-conv').textContent = fmt(b.conviction, 3);
  $('h-target').textContent = fmt(b.target_weight, 4);
  const pill = $('h-mode');
  pill.textContent = b.mode.toUpperCase();
  pill.classList.toggle('live', b.mode === 'live');

  $('b-regime').textContent = b.regime;
  $('b-conv').firstChild.textContent = fmt(b.conviction, 3);
  $('b-conv-bar').style.width = Math.min(100, Math.abs(b.conviction) * 100) + '%';
  $('b-target').textContent = fmt(b.target_weight, 4);
  $('b-current').textContent = fmt(b.current_weight, 4);
  $('b-dec').textContent = h.decisions_run;
  $('b-age').textContent = h.data_age_seconds > 1e6 ? '—' : h.data_age_seconds.toFixed(0) + 's';
}

function renderJournal() {
  const list = $('j-list');
  if (state.tab === 'trades') {
    const rows = state.journal.entries;
    if (!rows.length) { list.innerHTML = '<div class="empty">NO CLOSED TRADES</div>'; return; }
    list.innerHTML = rows.map((r) => `
      <div class="row" data-id="${r.id}">
        <span class="t">${shortTime(r.closed_at)}</span>
        <span class="s">${r.symbol} <span class="side ${r.side}">${r.side.toUpperCase()}</span></span>
        <span class="p ${r.net_pnl >= 0 ? 'pos' : 'neg'}">${pct(r.return_pct, 1)}</span>
        <span class="p ${r.net_pnl >= 0 ? 'pos' : 'neg'}">${signed(r.net_pnl)}</span>
      </div>`).join('');
  } else {
    const rows = state.journal.summaries.slice().reverse();
    if (!rows.length) { list.innerHTML = '<div class="empty">NO COMPLETED DAYS YET</div>'; return; }
    list.innerHTML = rows.map((s) => `
      <div class="row">
        <span class="t">${s.day}</span>
        <span class="s">${s.trades} trades · ${pct(s.win_rate, 0)} win</span>
        <span class="p ${s.net_pnl >= 0 ? 'pos' : 'neg'}">${pct(s.return_pct, 2)}</span>
        <span class="p ${s.net_pnl >= 0 ? 'pos' : 'neg'}">${signed(s.net_pnl)}</span>
      </div>`).join('');
  }
}

document.querySelectorAll('.tabs button').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tabs button').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    state.tab = btn.dataset.tab;
    renderJournal();
  });
});

/* -------------------------------------------------------- connections */

async function loadConnections() {
  const r = await fetch('/api/connections');
  const data = await r.json();
  $('k-path').textContent = data.store_path;
  $('k-count').textContent = data.exchanges.length;
  $('j-tg').textContent = 'telegram: ' + (data.telegram.configured ? 'linked' : 'not set');

  $('k-list').innerHTML = data.exchanges.length
    ? data.exchanges.map((e) => `
        <div class="conn">
          <span class="dot ${e.trade_enabled ? 'on' : ''}"></span>
          <span class="name">${e.label}</span>
          <span class="key">${e.api_key_masked}</span>
          <button data-key="${e.label}">×</button>
        </div>`).join('')
    : '<div class="empty">NO CONNECTIONS</div>';

  $('k-list').querySelectorAll('button').forEach((b) =>
    b.addEventListener('click', async () => {
      await fetch('/api/connections/' + encodeURIComponent(b.dataset.key), { method: 'DELETE' });
      loadConnections();
    }));
}

$('k-add').addEventListener('click', async () => {
  const payload = {
    exchange_id: $('k-ex').value.trim(),
    label: $('k-label').value.trim(),
    api_key: $('k-key').value,
    api_secret: $('k-secret').value,
    passphrase: $('k-pass').value,
    trade_enabled: $('k-trade').checked,
  };
  const r = await fetch('/api/connections', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const msg = $('k-msg');
  if (r.ok) {
    // Clear the secret fields immediately: no reason for key material to sit
    // in the DOM after it has been stored.
    ['k-key', 'k-secret', 'k-pass'].forEach((id) => ($(id).value = ''));
    msg.textContent = 'stored';
    loadConnections();
  } else {
    msg.textContent = (await r.json()).detail || 'failed';
  }
});

$('k-tg').addEventListener('click', async () => {
  const r = await fetch('/api/telegram/test', { method: 'POST' });
  $('k-msg').textContent = (await r.json()).message;
});

$('k-digest').addEventListener('click', async () => {
  const r = await fetch('/api/telegram/digest', { method: 'POST' });
  const d = await r.json();
  $('k-msg').textContent = d.ok ? 'digest sent' : 'telegram not configured';
});

/* ------------------------------------------------------------ stream */

function apply(snap) {
  syncNeurons(snap.neurons);
  renderHealth(snap.health);
  renderPnl(snap.pnl);
  renderBrain(snap.brain, snap.health);

  state.equityHistory.push(snap.pnl.equity);
  if (state.equityHistory.length > 400) state.equityHistory.shift();
  drawSpark();

  if (state.selected) {
    const found = snap.neurons.find((n) => n.id === state.selected);
    if (found) renderDetail(found);
  } else {
    renderDetail(null);
  }
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { state.connected = true; $('h-link').textContent = 'LIVE'; };
  ws.onmessage = (e) => apply(JSON.parse(e.data));
  ws.onclose = () => {
    state.connected = false;
    $('h-link').textContent = 'RECONNECTING';
    // The server is local; a drop usually means it restarted. Retry steadily
    // rather than backing off, so the page recovers as soon as it returns.
    setTimeout(connect, 1500);
  };
  ws.onerror = () => ws.close();
}

async function loadJournal() {
  try {
    const r = await fetch('/api/journal');
    state.journal = await r.json();
    renderJournal();
  } catch { /* server not up yet; the interval retries */ }
}

resize();
requestAnimationFrame(frame);
connect();
loadConnections();
loadJournal();
setInterval(loadJournal, 15000);
