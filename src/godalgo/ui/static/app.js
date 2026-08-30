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


/* ------------------------------------------------------ morphology
 *
 * Real neurons are not discs. Each has an irregular soma, a dense bush of
 * branching dendrites, and one longer axon that terminates in a synaptic
 * bouton against a neighbour.
 *
 * Morphology is generated once per neuron from a hash of its id and then only
 * transformed. Regenerating it per frame would make every cell shimmer and
 * writhe, which reads as noise rather than as biology -- a real cell has a
 * fixed shape and only its signalling changes.
 */

const TAU = Math.PI * 2;

/** Deterministic PRNG, so a given position always grows the same cell. */
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** One dendritic branch, grown recursively with tapering and drift. */
function growBranch(x, y, angle, length, width, depth, rand, out) {
  const segments = 3 + Math.floor(rand() * 2);
  const pts = [[x, y]];
  let cx = x, cy = y, ca = angle;

  for (let i = 0; i < segments; i++) {
    // Dendrites wander rather than running straight; the drift is what stops
    // them reading as spokes on a wheel.
    ca += (rand() - 0.5) * 0.75;
    const step = (length / segments) * (0.75 + rand() * 0.5);
    cx += Math.cos(ca) * step;
    cy += Math.sin(ca) * step;
    pts.push([cx, cy]);
  }
  out.push({ pts, width, tip: depth === 0 });

  if (depth > 0) {
    // Two children at a fork, each shorter and thinner -- the taper is most of
    // what makes a branch read as biological.
    const forks = rand() > 0.55 ? 2 : 1;
    for (let k = 0; k < forks; k++) {
      const spread = (k === 0 ? -1 : 1) * (0.4 + rand() * 0.6);
      growBranch(
        cx, cy, ca + spread,
        length * (0.52 + rand() * 0.22),
        width * 0.62,
        depth - 1, rand, out,
      );
    }
  }
}

function buildMorphology(id, radius) {
  const rand = mulberry32(hashString(id));

  // Soma: an irregular closed blob, not a circle.
  const soma = [];
  const lobes = 13 + Math.floor(rand() * 4);
  for (let i = 0; i < lobes; i++) {
    soma.push({
      a: (i / lobes) * TAU,
      // Gentle lobing. Heavier jitter turns the membrane into a spiky
      // star, which reads as a badly drawn polygon rather than a cell.
      r: radius * (0.90 + rand() * 0.17),
    });
  }

  // Dendritic arbour.
  const dendrites = [];
  // Fewer, finer, shorter branches. The arbour should frame the soma, not bury
  // it -- at higher densities the field turns into a mesh in which no
  // individual cell can be read or clicked.
  const primary = 4 + Math.floor(rand() * 3);
  const base = rand() * TAU;
  for (let i = 0; i < primary; i++) {
    const angle = base + (i / primary) * TAU + (rand() - 0.5) * 0.5;
    const start = radius * 0.94;
    growBranch(
      Math.cos(angle) * start, Math.sin(angle) * start, angle,
      radius * (1.0 + rand() * 0.9),
      Math.max(0.6, radius * 0.09),
      2, rand, dendrites,
    );
  }

  // The apical dendrite: one thick process, markedly longer than the rest,
  // leaving the pole opposite the axon. It is the single strongest cue that a
  // shape is a neuron rather than a generic blob, and pyramidal cells in
  // cortex are defined by it.
  const apicalAngle = rand() * TAU;
  growBranch(
    Math.cos(apicalAngle) * radius, Math.sin(apicalAngle) * radius, apicalAngle,
    radius * (2.4 + rand() * 1.1),
    Math.max(1.1, radius * 0.17),
    2, rand, dendrites,
  );

  // Nissl bodies -- the rough ER granules that give a real soma its mottled
  // interior. Without some internal texture the cell body renders as a glass
  // bead, which is the single strongest cue that it is not biological.
  const nissl = [];
  const grains = 5 + Math.floor(rand() * 5);
  for (let i = 0; i < grains; i++) {
    const a = rand() * TAU;
    const rr = rand() * radius * 0.62;
    nissl.push({
      dx: Math.cos(a) * rr,
      dy: Math.sin(a) * rr,
      r: radius * (0.07 + rand() * 0.09),
      o: 0.12 + rand() * 0.16,
    });
  }

  return {
    soma,
    dendrites,
    nissl,
    // Real somas are rarely round -- most are ovoid or pyramidal, with the
    // axon leaving from one pole.
    stretch: 1.0 + rand() * 0.38,
    orient: rand() * TAU,
    nucleus: {
      dx: (rand() - 0.5) * radius * 0.34,
      dy: (rand() - 0.5) * radius * 0.34,
      r: radius * (0.28 + rand() * 0.10),
    },
    spacing: 0.85 + rand() * 0.4,
    phase: rand() * TAU,
  };
}

/** Trace the membrane, stretched along the cell's own axis. */

/** Render a cell's static anatomy into an offscreen sprite.
 *
 * A neuron's bloom, membrane and nucleus are all radial gradients, and
 * createRadialGradient is expensive -- rebuilding them for every cell on every
 * frame costs more than everything else in the render combined. Nothing about
 * that anatomy changes between frames, so it is drawn once here and blitted
 * thereafter; only the signalling (axon pulses, breathing alpha, selection) is
 * recomputed live.
 */
function buildSprite(node) {
  const [r, g, b] = COLOR[node.data.state] || COLOR.open;
  const m = node.morph;
  const dpr = window.devicePixelRatio || 1;
  const pad = Math.ceil(node.r * 4.4);
  const size = pad * 2;

  const off = document.createElement('canvas');
  off.width = off.height = Math.ceil(size * dpr);
  const c = off.getContext('2d');
  c.scale(dpr, dpr);
  c.translate(pad, pad);

  // Ambient bloom.
  const halo = node.r * 3.2;
  const bloom = c.createRadialGradient(0, 0, node.r * 0.4, 0, 0, halo);
  bloom.addColorStop(0, `rgba(${r},${g},${b},0.17)`);
  bloom.addColorStop(0.45, `rgba(${r},${g},${b},0.05)`);
  bloom.addColorStop(1, `rgba(${r},${g},${b},0)`);
  c.fillStyle = bloom;
  c.beginPath(); c.arc(0, 0, halo, 0, TAU); c.fill();

  // Dendrites.
  c.lineCap = 'round';
  c.lineJoin = 'round';
  for (const branch of m.dendrites) {
    c.beginPath();
    const first = branch.pts[0];
    c.moveTo(first[0], first[1]);
    for (let i = 1; i < branch.pts.length; i++) {
      const prev = branch.pts[i - 1], cur = branch.pts[i];
      c.quadraticCurveTo(prev[0], prev[1], (prev[0] + cur[0]) / 2, (prev[1] + cur[1]) / 2);
    }
    c.strokeStyle = `rgba(${r},${g},${b},0.24)`;
    c.lineWidth = branch.width;
    c.stroke();

    if (branch.tip) {
      const last = branch.pts[branch.pts.length - 1];
      c.fillStyle = `rgba(${r},${g},${b},0.34)`;
      c.beginPath(); c.arc(last[0], last[1], branch.width * 0.6, 0, TAU); c.fill();
    }
  }

  // Membrane path in local coordinates.
  const cos = Math.cos(m.orient), sin = Math.sin(m.orient);
  const trace = () => {
    const at = (p) => {
      const lx = Math.cos(p.a) * p.r * m.stretch;
      const ly = Math.sin(p.a) * p.r;
      return [lx * cos - ly * sin, lx * sin + ly * cos];
    };
    c.beginPath();
    for (let i = 0; i <= m.soma.length; i++) {
      const [px, py] = at(m.soma[i % m.soma.length]);
      const [qx, qy] = at(m.soma[(i + 1) % m.soma.length]);
      if (i === 0) c.moveTo(px, py);
      c.quadraticCurveTo(px, py, (px + qx) / 2, (py + qy) / 2);
    }
    c.closePath();
  };

  // Cytoplasm.
  trace();
  const body = c.createRadialGradient(
    -node.r * 0.15, -node.r * 0.18, node.r * 0.25, 0, 0, node.r * 1.25,
  );
  body.addColorStop(0, `rgba(${Math.min(255, r + 34)},${Math.min(255, g + 34)},${Math.min(255, b + 34)},0.86)`);
  body.addColorStop(0.68, `rgba(${r * 0.86},${g * 0.86},${b * 0.86},0.80)`);
  body.addColorStop(1, `rgba(${r * 0.42},${g * 0.42},${b * 0.42},0.74)`);
  c.fillStyle = body;
  c.fill();

  // Nissl granules, clipped to the membrane.
  c.save();
  trace();
  c.clip();
  for (const grain of m.nissl) {
    c.fillStyle = `rgba(${r * 0.35},${g * 0.35},${b * 0.35},${grain.o})`;
    c.beginPath();
    c.ellipse(grain.dx, grain.dy, grain.r * 1.4, grain.r, m.orient, 0, TAU);
    c.fill();
  }
  c.restore();

  // Membrane rim.
  trace();
  c.strokeStyle = `rgba(${Math.min(255, r + 70)},${Math.min(255, g + 70)},${Math.min(255, b + 70)},0.9)`;
  c.lineWidth = 1.2;
  c.stroke();

  // Nucleus and nucleolus.
  const nuc = m.nucleus;
  const nucleus = c.createRadialGradient(nuc.dx, nuc.dy, 0, nuc.dx, nuc.dy, nuc.r * 1.5);
  nucleus.addColorStop(0, `rgba(${Math.min(255, r + 90)},${Math.min(255, g + 90)},${Math.min(255, b + 90)},0.34)`);
  nucleus.addColorStop(0.7, `rgba(${Math.min(255, r + 40)},${Math.min(255, g + 40)},${Math.min(255, b + 40)},0.14)`);
  nucleus.addColorStop(1, `rgba(${r},${g},${b},0)`);
  c.fillStyle = nucleus;
  c.beginPath();
  c.ellipse(nuc.dx, nuc.dy, nuc.r * 1.5 * m.stretch, nuc.r * 1.5, m.orient, 0, TAU);
  c.fill();

  c.fillStyle = `rgba(${r * 0.32},${g * 0.32},${b * 0.32},0.34)`;
  c.beginPath(); c.arc(nuc.dx, nuc.dy, nuc.r * 0.3, 0, TAU); c.fill();

  node.sprite = off;
  node.spritePad = pad;
  node.spriteSize = size;
  node.spriteState = node.data.state;
}

function somaPath(node, scale) {
  const m = node.morph;
  const cos = Math.cos(m.orient), sin = Math.sin(m.orient);

  const at = (p) => {
    // Build the point in the cell's frame, stretch it, then rotate back.
    const lx = Math.cos(p.a) * p.r * scale * m.stretch;
    const ly = Math.sin(p.a) * p.r * scale;
    return [node.x + lx * cos - ly * sin, node.y + lx * sin + ly * cos];
  };

  ctx.beginPath();
  for (let i = 0; i <= m.soma.length; i++) {
    const [px, py] = at(m.soma[i % m.soma.length]);
    const [qx, qy] = at(m.soma[(i + 1) % m.soma.length]);
    if (i === 0) ctx.moveTo(px, py);
    // Quadratic through the midpoint keeps the membrane smooth rather than
    // faceted, which is what separates a cell from a polygon.
    ctx.quadraticCurveTo(px, py, (px + qx) / 2, (py + qy) / 2);
  }
  ctx.closePath();
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
  // Sprites are rasterised at the current device pixel ratio; a move between
  // displays would otherwise leave them soft.
  for (const n of state.neurons.values()) n.sprite = null;
}
window.addEventListener('resize', resize);

// A window resize listener alone is not enough: the canvas is a flex child, so
// it also changes size when the grid reflows or a sibling panel grows, with no
// window event. Stale dimensions let the soft walls sit outside the visible
// area and neurons get clipped at the edge.
if (window.ResizeObserver) new ResizeObserver(resize).observe(canvas);

/** Radius from notional, on a square root so a 100x larger position is 10x
 *  wider rather than 100x -- otherwise one big trade fills the panel.
 *
 *  Scaled down as the field fills. A cell needs roughly three times its soma
 *  radius of clear space for its arbour; at fixed size a busy cluster mats the
 *  dendrites into an indistinct felt where no individual cell can be read. */
function radiusFor(n, count) {
  const base = Math.sqrt(Math.max(n.notional, 1)) * 0.42;
  const crowding = Math.min(1, Math.sqrt(24 / Math.max(count, 1)));
  const scaled = base * (0.55 + 0.45 * crowding);
  return Math.max(4.5, Math.min(24, scaled));
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
      const ring = 0.25 + Math.random() * 0.7;
      node = {
        x: W / 2 + Math.cos(angle) * W * 0.44 * ring,
        y: H / 2 + Math.sin(angle) * H * 0.42 * ring,
        vx: Math.cos(angle) * 0.3,
        vy: Math.sin(angle) * 0.3,
        born: performance.now(),
        data,
      };
      state.neurons.set(data.id, node);
    } else {
      node.data = data;
    }
    const r = radiusFor(data, list.length);
    // Rebuild only when the radius actually changes, so the cell keeps its
    // identity frame to frame.
    if (node.r !== r || !node.morph) {
      node.r = r;
      node.morph = buildMorphology(data.id, r);
      node.sprite = null;
    }
    // State drives colour, so a close repaints the cell.
    if (!node.sprite || node.spriteState !== data.state) buildSprite(node);
  }
  for (const id of [...state.neurons.keys()]) {
    if (!seen.has(id)) state.neurons.delete(id);
  }
  const total = state.lastTotal || list.length;
  $('c-count').textContent = total > list.length
    ? `${list.length} of ${total} positions`
    : `${list.length} position${list.length === 1 ? '' : 's'}`;
  $('c-empty').style.display = list.length ? 'none' : 'flex';
}

function step() {
  const nodes = [...state.neurons.values()];
  if (!nodes.length) return;
  const cx = W / 2, cy = H / 2;

  // Elliptical containment, matched to the panel's own proportions. A circular
  // bound sized to the shorter side leaves the whole horizontal margin of a
  // wide panel empty -- the cluster floats as a disc in a letterbox.
  const spreadX = W * 0.46;
  const spreadY = H * 0.44;

  for (const a of nodes) {
    const dx = a.x - cx, dy = a.y - cy;
    // Normalised radius: 1.0 is on the ellipse, above it is outside.
    const norm = Math.hypot(dx / spreadX, dy / spreadY);
    if (norm > 1) {
      const pull = (norm - 1) * 0.9;
      const dist = Math.hypot(dx, dy) || 0.01;
      a.vx -= (dx / dist) * pull;
      a.vy -= (dy / dist) * pull;
    }
    // Brownian jitter: the "alive" part.
    a.vx += (Math.random() - 0.5) * 0.05;
    a.vy += (Math.random() - 0.5) * 0.05;
  }

  // Repulsion at a range scaled to how much room each node can claim, so a
  // sparse cluster spreads out and a crowded one packs without overlapping.
  // Cells need clearance for their arbours, so the repulsion floor scales with
  // the soma radius rather than being a flat pixel count.
  const room = Math.sqrt((W * H) / nodes.length);
  const reach = Math.max(46, Math.min(160, room * 0.82));

  // Same spatial bucketing as the axon pass. Repulsion is naturally O(n^2) and
  // runs every frame, so on a busy field it costs more than the render.
  const maxMin = reach * 1.25 + 60;
  const bucketSize = maxMin;
  const grid = new Map();
  for (const n of nodes) {
    const key = `${Math.floor(n.x / bucketSize)},${Math.floor(n.y / bucketSize)}`;
    let bucket = grid.get(key);
    if (!bucket) grid.set(key, (bucket = []));
    bucket.push(n);
  }

  for (const a of nodes) {
    const gx = Math.floor(a.x / bucketSize), gy = Math.floor(a.y / bucketSize);
    for (let ox = -1; ox <= 1; ox++) {
      for (let oy = -1; oy <= 1; oy++) {
        const bucket = grid.get(`${gx + ox},${gy + oy}`);
        if (!bucket) continue;
        for (const b of bucket) {
          // Each unordered pair is handled once; both nodes are pushed here.
          if (b === a || b.data.id <= a.data.id) continue;
          let dx = b.x - a.x, dy = b.y - a.y;
          const d2 = dx * dx + dy * dy;
          // Per-cell spacing preference. A single uniform separation packs the
          // field into a visible lattice, which no tissue looks like.
          const min = a.r + b.r + reach * 0.5 * (a.morph.spacing + b.morph.spacing);
          if (d2 >= min * min) continue;
          const d = Math.sqrt(d2) || 0.01;
          const push = ((min - d) / min) * 0.16;
          dx /= d; dy /= d;
          a.vx -= dx * push; a.vy -= dy * push;
          b.vx += dx * push; b.vy += dy * push;
        }
      }
    }
  }

  for (const a of nodes) {
    a.vx *= 0.955; a.vy *= 0.955;          // damping keeps motion calm
    a.x += a.vx; a.y += a.vy;
    // Soft walls. Padding covers the dendritic arbour and the label beneath,
    // not just the soma -- clipping a cell's branches at the edge looks like a
    // rendering fault rather than a boundary.
    const pad = a.r * 2.4 + 14;
    if (a.x < pad)      { a.x = pad;     a.vx = Math.abs(a.vx) * 0.6; }
    if (a.x > W - pad)  { a.x = W - pad; a.vx = -Math.abs(a.vx) * 0.6; }
    if (a.y < pad)      { a.y = pad;     a.vy = Math.abs(a.vy) * 0.6; }
    if (a.y > H - pad)  { a.y = H - pad; a.vy = -Math.abs(a.vy) * 0.6; }
  }
}

function draw(now) {
  ctx.clearRect(0, 0, W, H);

  // Backdrop grid — cheap depth, drawn under everything.
  ctx.strokeStyle = 'rgba(16,24,39,0.5)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < W; x += 44) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
  for (let y = 0; y < H; y += 44) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
  ctx.stroke();

  const nodes = [...state.neurons.values()];
  const t = now / 1000;

  // ---- pass 1: axons ------------------------------------------------------
  // Drawn before the cells so a process appears to pass behind a soma rather
  // than across it. An axon leaves one membrane, arcs, and ends in a synaptic
  // bouton pressed against its target.
  const reachMax = 175;
  const cutoff = reachMax * 0.86;
  const cutoffSq = cutoff * cutoff;

  // Spatial hash. Comparing every pair is O(n^2) and dominates the frame once
  // the field is busy -- which is exactly when the render must stay smooth.
  // Bucketing at the interaction radius means each cell only tests the nine
  // cells that could possibly be in range.
  const cell = cutoff;
  const buckets = new Map();
  for (const n of nodes) {
    const key = `${Math.floor(n.x / cell)},${Math.floor(n.y / cell)}`;
    let bucket = buckets.get(key);
    if (!bucket) buckets.set(key, (bucket = []));
    bucket.push(n);
  }

  // Each cell keeps only its nearest few partners. Real neurons synapse onto a
  // handful of targets, not onto everything within reach -- and drawing every
  // near pair produces a felt of lines that hides the cells underneath.
  const MAX_LINKS = 3;
  const links = new Map();

  for (const a of nodes) {
    const gx = Math.floor(a.x / cell), gy = Math.floor(a.y / cell);
    const near = [];
    for (let ox = -1; ox <= 1; ox++) {
      for (let oy = -1; oy <= 1; oy++) {
        const bucket = buckets.get(`${gx + ox},${gy + oy}`);
        if (!bucket) continue;
        for (const b of bucket) {
          if (b === a) continue;
          const dx = b.x - a.x, dy = b.y - a.y;
          const d2 = dx * dx + dy * dy;
          if (d2 > cutoffSq || d2 < 1) continue;
          near.push({ b, d2 });
        }
      }
    }
    near.sort((p, q) => p.d2 - q.d2);
    for (const { b } of near.slice(0, MAX_LINKS)) {
      const key = a.data.id < b.data.id
        ? a.data.id + '|' + b.data.id
        : b.data.id + '|' + a.data.id;
      if (!links.has(key)) links.set(key, [a, b]);
    }
  }

  for (const [key, [a, b]] of links) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const d = Math.hypot(dx, dy) || 1;
    const strength = 1 - d / reachMax;
    const sameSymbol = a.data.symbol === b.data.symbol;
    const ux = dx / d, uy = dy / d;

    const x1 = a.x + ux * a.r * 0.95, y1 = a.y + uy * a.r * 0.95;
    const x2 = b.x - ux * (b.r * 0.95 + 3), y2 = b.y - uy * (b.r * 0.95 + 3);

    const h = hashString(key);
    const bow = ((h % 100) / 100 - 0.5) * d * 0.28;
    const mx = (x1 + x2) / 2 - uy * bow, my = (y1 + y2) / 2 + ux * bow;

    const [ar, ag, ab] = COLOR[a.data.state] || COLOR.open;
    const alpha = strength * (sameSymbol ? 0.62 : 0.28);

    // Myelin sheath under a brighter axoplasm core.
    ctx.strokeStyle = sameSymbol
      ? `rgba(255,170,80,${alpha * 0.35})`
      : `rgba(120,190,220,${alpha * 0.3})`;
    ctx.lineWidth = 2.4 * strength + 0.6;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(mx, my, x2, y2); ctx.stroke();

    ctx.strokeStyle = sameSymbol
      ? `rgba(255,200,140,${alpha})`
      : `rgba(150,220,245,${alpha * 0.85})`;
    ctx.lineWidth = Math.max(0.5, 1.0 * strength);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(mx, my, x2, y2); ctx.stroke();

    // Synaptic bouton: the terminal swelling against the target membrane.
    ctx.fillStyle = `rgba(${ar},${ag},${ab},${alpha * 1.8})`;
    ctx.beginPath(); ctx.arc(x2, y2, 1.5 + strength * 1.4, 0, TAU); ctx.fill();

    // Action potential travelling the axon.
    if (strength > 0.3) {
      const speed = 0.45 + (h % 50) / 120;
      const u = (t * speed) % 1;
      const iu = 1 - u;
      const px = iu * iu * x1 + 2 * iu * u * mx + u * u * x2;
      const py = iu * iu * y1 + 2 * iu * u * my + u * u * y2;
      const fade = Math.sin(u * Math.PI) * strength;
      ctx.globalCompositeOperation = 'lighter';
      ctx.fillStyle = `rgba(255,255,255,${0.55 * fade})`;
      ctx.beginPath(); ctx.arc(px, py, 1.8, 0, TAU); ctx.fill();
      ctx.fillStyle = `rgba(${ar},${ag},${ab},${0.35 * fade})`;
      ctx.beginPath(); ctx.arc(px, py, 4.4, 0, TAU); ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
    }
  }

  // ---- pass 2: cells ------------------------------------------------------
  // Pre-rendered sprites. Only the breathing alpha varies per frame; the
  // anatomy itself is fixed, so redrawing it would be work for no change.
  for (const n of nodes) {
    if (!n.sprite) continue;
    const breathe = n.data.is_open
      ? 0.78 + 0.22 * Math.sin(t * 1.8 + n.morph.phase)
      : 1;
    ctx.globalAlpha = breathe;
    ctx.drawImage(n.sprite, n.x - n.spritePad, n.y - n.spritePad, n.spriteSize, n.spriteSize);
    ctx.globalAlpha = 1;
  }

  // ---- pass 3: selection and labels ---------------------------------------
  for (const n of nodes) {
    if (state.selected === n.data.id) {
      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 8, 0, TAU); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (n.r <= 9) continue;
    const ly = n.y + n.r + 13;
    if (ly >= H - 2) continue;
    ctx.fillStyle = 'rgba(234,242,255,0.72)';
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(n.data.symbol.split('/')[0], n.x, ly);
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
  state.lastTotal = snap.pnl.total_count;
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
