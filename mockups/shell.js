/* Shared shell + chart helpers for the tradebot redesign mockups (static, no build step). */
(function () {
  const ICONS = {
    chart: 'M3 3v18h18M7 15l4-5 3 3 5-7',
    screener: 'M3 5h18l-7 8v6l-4 2v-8z',
    scanner: 'M12 3a9 9 0 1 0 9 9M12 7.5a4.5 4.5 0 1 0 4.5 4.5M12 12l7-7',
    flask: 'M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3M7.5 15h9',
    dna: 'M8 3c0 5 8 5 8 9s-8 4-8 9M16 3c0 5-8 5-8 9s8 4 8 9M9 7h6M9 17h6',
    pairs: 'M4 8h13l-3-3M20 16H7l3 3',
    wallet: 'M3 7a2 2 0 0 1 2-2h13v4M3 7v11a2 2 0 0 0 2 2h15V9H5a2 2 0 0 1-2-2M16 14.5h.01',
    news: 'M5 4h12v16H7a2 2 0 0 1-2-2zM17 8h3v10a2 2 0 0 1-2 2M8 8h6M8 12h6M8 16h4',
    settings: 'M4 7h10M18 7h2M4 17h2M10 17h10M16 7m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0M8 17m-2 0a2 2 0 1 0 4 0a2 2 0 1 0-4 0',
    info: 'M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18M12 11v5M12 8h.01',
    power: 'M12 3v8M6.3 6.3a8 8 0 1 0 11.4 0',
    plus: 'M12 5v14M5 12h14', close: 'M6 6l12 12M18 6L6 18', search: 'M11 4a7 7 0 1 0 0 14a7 7 0 0 0 0-14M16 16l5 5',
    star: 'M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z',
    cursor: 'M5 3l14 8-6 2-2 6z', trend: 'M4 19L20 5M4 19m-1.5 0a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0-3 0M20 5m-1.5 0a1.5 1.5 0 1 0 3 0a1.5 1.5 0 1 0-3 0',
    fib: 'M4 5h16M4 10h16M4 15h16M4 20h16', text: 'M5 6h14M12 6v13M9 19h6', vline: 'M12 3v18M9 6l3-3 3 3',
    ruler: 'M3 17L17 3l4 4L7 21zM8 12l2 2M11 9l2 2M14 6l2 2', trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13',
    layers: 'M12 3l9 5-9 5-9-5zM3 13l9 5 9-5', code: 'M8 7l-5 5 5 5M16 7l5 5-5 5M14 4l-4 16',
    play: 'M7 4l13 8-13 8z', stop: 'M6 6h12v12H6z', download: 'M12 4v11M7 11l5 5 5-5M5 20h14',
    chev: 'M6 9l6 6 6-6', chevr: 'M9 6l6 6-6 6', folder: 'M3 6a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z',
    refresh: 'M20 11a8 8 0 1 0-2 6M20 4v7h-7', lock: 'M6 11h12v9H6zM8 11V8a4 4 0 0 1 8 0v3', key: 'M8 11a4 4 0 1 0 0 .1M11 13l9 8M16 18l2-2',
    db: 'M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3',
    cpu: 'M7 7h10v10H7zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4', sidebar: 'M3 5h18v14H3zM15 5v14',
    moon: 'M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z', check: 'M5 12l5 5L20 7',
    alert: 'M12 3l10 18H2zM12 10v5M12 18h.01', ext: 'M14 4h6v6M20 4l-9 9M18 14v5H5V6h5', save: 'M5 3h12l4 4v14H5zM8 3v6h8V3M8 21v-7h8v7',
    filter: 'M4 6h16M7 12h10M10 18h4', grid: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z', clock: 'M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18M12 7v5l3 2',
    bolt: 'M13 3L5 14h6l-1 7 8-11h-6z', copy: 'M8 8h12v12H8zM4 16V4h12', eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0-6 0',
    undo: 'M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3', merge: 'M6 4v6a6 6 0 0 0 6 6h6M18 12l3 4-3 4', bell: 'M6 16V11a6 6 0 0 1 12 0v5l2 2H4zM10 21h4',
    pin: 'M12 17v5M8 3h8l-1 6 3 4H6l3-4z', dots: 'M5 12h.01M12 12h.01M19 12h.01', hand: 'M8 13V6a1.5 1.5 0 0 1 3 0v6M11 11V4.5a1.5 1.5 0 0 1 3 0V12M14 12V6.5a1.5 1.5 0 0 1 3 0V15a6 6 0 0 1-6 6h-1a6 6 0 0 1-5-3l-2-4a1.5 1.5 0 0 1 2.5-1.5L8 14'
  };
  const sprite = '<svg xmlns="http://www.w3.org/2000/svg" style="display:none">' +
    Object.entries(ICONS).map(([k, d]) => `<symbol id="i-${k}" viewBox="0 0 24 24"><path d="${d}"/></symbol>`).join('') + '</svg>';
  const ico = (n, cls = '') => `<svg class="i ${cls}"><use href="#i-${n}"/></svg>`;
  const fillIcons = (root) => root.querySelectorAll('i[data-i]').forEach(e => { e.outerHTML = ico(e.dataset.i); });

  const NAV = [
    ['Markets', [['chart', 'Chart', 'chart.html', 'G C'], ['screener', 'Screener', 'screener.html'], ['scanner', 'Scanner', 'scanner.html'], ['pairs', 'Pairs', 'pairs.html'], ['news', 'News', 'news.html']]],
    ['Strategies', [['flask', 'Backtest', 'backtest.html'], ['dna', 'Evolve', 'evolve.html'], ['code', 'Scripts', 'editor.html']]],
    ['Account', [['wallet', 'Portfolio', 'portfolio.html']]]
  ];
  const MODES = { dry: ['DRY RUN', ''], sandbox: ['SANDBOX', 'sandbox'], live: ['LIVE', 'live'] };

  function build() {
    const b = document.body, page = b.dataset.page, title = b.dataset.title || '';
    const theme = localStorage.getItem('tb-theme') || new URLSearchParams(location.search).get('theme') || 'graphite';
    document.documentElement.dataset.theme = theme;
    document.body.insertAdjacentHTML('afterbegin', sprite);
    const view = document.getElementById('view');
    const topT = document.getElementById('top');
    const app = document.createElement('div');
    app.className = 'app';
    const rail = NAV.map(([g, items]) => `<div class="navgroup">${g}</div>` + items.map(([i, n, h, k]) =>
      `<a class="nav" href="${h}" ${h === page + '.html' ? 'aria-current="page"' : ''}>${ico(i)}<span>${n}</span>${k ? `<kbd>${k}</kbd>` : ''}</a>`).join('')).join('');
    app.innerHTML = `
      <nav class="rail" aria-label="Main">
        <div class="brand"><i></i><b>tradebot</b><span>v1.12</span></div>${rail}<div class="grow"></div>
        <a class="nav" href="settings.html" ${page === 'settings' ? 'aria-current="page"' : ''}>${ico('settings')}<span>Settings</span><kbd>,</kbd></a>
        <a class="nav" href="dialogs.html">${ico('layers')}<span>Dialogs &amp; states</span></a>
        <a class="nav" href="index.html">${ico('grid')}<span>All mockups</span></a>
      </nav>
      <header class="top"><h1>${title}</h1><div class="row sp" id="topslot"></div></header>
      <main class="main" id="mainslot"></main>
      <footer class="stat">
        <button class="mode" id="modechip" title="Mockup: click to cycle dry run, sandbox and live"></button>
        <button class="kill" id="kill" aria-pressed="false" title="Stops every running live bot at its next check">${ico('power')}<span>Kill switch</span></button>
        <span class="item" id="stmsg"></span><span class="sp"></span>
        <span class="item"><span class="dot"></span>Binance · Yahoo</span>
        <span class="item">${ico('db')}Cache 412 MB</span>
        <span class="item">${ico('cpu')}CPU · 8 threads</span><button class="btn ghost sm" id="tt" title="Mockup only: switch theme" style="height:20px">${ico('moon')}<span></span></button>
      </footer>`;
    if (topT) app.querySelector('#topslot').append(topT.content.cloneNode(true));
    app.querySelector('#mainslot').append(view);
    b.append(app);
        const tt = document.getElementById('tt'); const setT = () => tt.querySelector('span').textContent = document.documentElement.dataset.theme === 'paper' ? 'Graphite' : 'Paper';
    setT(); tt.onclick = () => { const n = document.documentElement.dataset.theme === 'paper' ? 'graphite' : 'paper'; document.documentElement.dataset.theme = n; localStorage.setItem('tb-theme', n); setT(); redraw(); };
    let mode = b.dataset.mode || 'dry'; const chip = document.getElementById('modechip');
    const setM = () => { const [t, c] = MODES[mode]; chip.className = 'mode ' + c; chip.innerHTML = `<span class="dot ${mode === 'live' ? 'live' : mode === 'sandbox' ? 'warn' : 'off'}"></span>${t}`; app.style.boxShadow = mode === 'live' ? 'inset 0 2px 0 var(--live)' : ''; };
    setM(); chip.onclick = () => { mode = mode === 'dry' ? 'sandbox' : mode === 'sandbox' ? 'live' : 'dry'; setM(); };
    const kill = document.getElementById('kill');
    kill.onclick = () => { const on = kill.getAttribute('aria-pressed') !== 'true'; kill.setAttribute('aria-pressed', on); kill.querySelector('span').textContent = on ? 'Kill switch ON' : 'Kill switch'; banner(on); };
    if (b.dataset.kill) kill.click();
    function banner(on) {
      let el = document.getElementById('killbanner');
      if (!on) return el && el.remove();
      const t = document.createElement('div'); t.id = 'killbanner';
      t.style.cssText = 'position:absolute;left:0;right:0;top:0;z-index:40;background:var(--live-bg);color:#fff;padding:6px 16px;font-weight:500;display:flex;gap:10px;align-items:center';
      t.innerHTML = `${ico('power')}Kill switch is on. Every live bot stops at its next check and no new orders go out.<span class="sp" style="flex:1"></span><button class="btn sm" style="background:rgba(0,0,0,.25);border-color:rgba(255,255,255,.4);color:#fff" onclick="document.getElementById('kill').click()">Turn off</button>`;
      document.getElementById('mainslot').style.position = 'relative'; document.getElementById('mainslot').prepend(t);
    }
    document.getElementById('stmsg').textContent = b.dataset.status || '';
    fillIcons(document);
    document.querySelectorAll('[data-sort]').forEach(() => {});
    redraw();
  }

  /* ---------- charts ---------- */
  const NS = 'http://www.w3.org/2000/svg';
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const rng = (s) => () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const randn = (r) => { let u = 0; for (let i = 0; i < 6; i++) u += r(); return (u - 3) / Math.sqrt(.5); };
  function walk(seed, n, start = 100, vol = .012, drift = .0003) {
    const r = rng(seed), out = []; let p = start;
    for (let i = 0; i < n; i++) { const o = p, ch = randn(r) * vol + drift + Math.sin(i / 23) * vol * .25; const c = o * (1 + ch);
      const hi = Math.max(o, c) * (1 + Math.abs(randn(r)) * vol * .5), lo = Math.min(o, c) * (1 - Math.abs(randn(r)) * vol * .5);
      out.push({ o, h: hi, l: lo, c, v: 40 + r() * 60 + Math.abs(ch) * 4000 }); p = c; }
    return out;
  }
  const ema = (a, k) => { const m = 2 / (k + 1); let e = a[0]; return a.map(x => (e = x * m + e * (1 - m))); };
  const el = (t, a = {}, p) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (p) p.appendChild(e); return e; };
  const fmt = (v, d = 2) => v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
  function svgFor(host) { host.querySelector('svg.chart')?.remove(); const s = el('svg', { class: 'chart' }); host.prepend(s); const w = host.clientWidth, h = host.clientHeight; s.setAttribute('viewBox', `0 0 ${w} ${h}`); return [s, w, h]; }
  const scale = (d0, d1, r0, r1) => (v) => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
  function axisY(s, w, y, lo, hi, ticks, padR, f) { for (let i = 0; i <= ticks; i++) { const v = lo + (hi - lo) * i / ticks, yy = y(v); el('line', { x1: 0, x2: w - padR, y1: yy, y2: yy, class: 'g' }, s); el('text', { x: w - padR + 8, y: yy + 4 }, s).textContent = f(v); } }
  function xlabels(s, x, n, h, count, labels) { for (let i = 0; i < count; i++) { const idx = Math.round((n - 1) * (i + .5) / count); el('text', { x: x(idx), y: h - 6, 'text-anchor': 'middle' }, s).textContent = labels[i % labels.length]; } }

  const MONTHS = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function candles(host) {
    const o = JSON.parse(host.dataset.candles || '{}'); const n = o.n || 90, data = walk(o.seed || 7, n, o.start || 62000, o.vol || .011, o.drift || .0004);
    const [s, w, h] = svgFor(host); const padR = 62, padB = 22, volH = o.vol === false ? 0 : h * .14, subH = o.sub ? h * .2 : 0;
    const top = 8, mainB = h - padB - volH - subH - 4, up = css('--up'), dn = css('--dn');
    const hi = Math.max(...data.map(d => d.h)), lo = Math.min(...data.map(d => d.l)), pad = (hi - lo) * .06;
    const y = scale(hi + pad, lo - pad, top, mainB), x = (i) => 8 + (i + .5) * (w - padR - 16) / n, cw = Math.max(2, (w - padR - 16) / n * .66);
    axisY(s, w, y, lo - pad, hi + pad, 5, padR, v => fmt(v, 0));
    xlabels(s, x, n, h, 6, MONTHS);
    const vmax = Math.max(...data.map(d => d.v));
    if (volH) data.forEach((d, i) => { const hh = d.v / vmax * volH; el('rect', { x: x(i) - cw / 2, y: h - padB - hh, width: cw, height: hh, fill: d.c >= d.o ? up : dn, opacity: .28 }, s); });
    if (o.bands) { const m = ema(data.map(d => d.c), 20); const pts = (k) => m.map((v, i) => `${x(i)},${y(v * (1 + k))}`).join(' ');
      el('polygon', { points: m.map((v, i) => `${x(i)},${y(v * 1.03)}`).join(' ') + ' ' + m.map((v, i) => `${x(n - 1 - i)},${y(m[n - 1 - i] * .97)}`).join(' '), fill: css('--accent'), opacity: .07 }, s); }
    data.forEach((d, i) => { const c = d.c >= d.o ? up : dn; el('line', { x1: x(i), x2: x(i), y1: y(d.h), y2: y(d.l), stroke: c, 'stroke-width': 1.2 }, s);
      const t = Math.min(y(d.o), y(d.c)), hh = Math.max(1.5, Math.abs(y(d.o) - y(d.c))); el('rect', { x: x(i) - cw / 2, y: t, width: cw, height: hh, fill: d.c >= d.o ? 'none' : c, stroke: c, 'stroke-width': 1.2, rx: 1 }, s); });
    (o.ema || []).forEach((k, j) => { const m = ema(data.map(d => d.c), k); el('polyline', { points: m.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: css('--s' + (j * 2 + 3 > 6 ? 3 : j * 2 + 3)), 'stroke-width': 1.5 }, s); });
    if (o.signals) { const f = ema(data.map(d => d.c), 9), sl = ema(data.map(d => d.c), 21);
      for (let i = 2; i < n; i++) { const a = f[i - 1] - sl[i - 1], b = f[i] - sl[i];
        if (a <= 0 && b > 0) { const yy = y(data[i].l) + 12; el('path', { d: `M${x(i)} ${yy - 8}l5 8h-10z`, fill: up }, s); }
        if (a >= 0 && b < 0) { const yy = y(data[i].h) - 12; el('path', { d: `M${x(i)} ${yy + 8}l5 -8h-10z`, fill: dn }, s); } } }
    if (subH) { const base = h - padB - volH - subH + 6, r = rng(11); let v = 50; const rs = data.map(() => (v = Math.max(18, Math.min(82, v + (r() - .5) * 16)))); const yy = scale(100, 0, base, base + subH - 10);
      [30, 70].forEach(t => { el('line', { x1: 0, x2: w - padR, y1: yy(t), y2: yy(t), class: 'g' }, s); el('text', { x: w - padR + 8, y: yy(t) + 4 }, s).textContent = t; });
      el('polyline', { points: rs.map((q, i) => `${x(i)},${yy(q)}`).join(' '), fill: 'none', stroke: css('--accent'), 'stroke-width': 1.5 }, s); el('text', { x: 10, y: base + 8, class: 'lbl' }, s).textContent = 'RSI 14'; }
    const last = data[n - 1], ly = y(last.c); el('line', { x1: 0, x2: w - padR, y1: ly, y2: ly, stroke: last.c >= last.o ? up : dn, 'stroke-dasharray': '2 3', opacity: .7 }, s);
    el('rect', { x: w - padR + 2, y: ly - 9, width: padR - 4, height: 18, rx: 3, fill: last.c >= last.o ? up : dn }, s); el('text', { x: w - padR + 8, y: ly + 4, fill: '#0b0b0a', style: 'fill:#0b0b0a;font-weight:600' }, s).textContent = fmt(last.c, 0);
    if (o.draw) { // saved drawings: trendline, fib, note
      const a = Math.floor(n * .28), b = Math.floor(n * .62), ya = y(data[a].l), yb = y(data[b].h);
      el('line', { x1: x(a), y1: ya, x2: x(b), y2: yb, stroke: css('--s3'), 'stroke-width': 1.6 }, s);
      [[a, ya], [b, yb]].forEach(([i, yy]) => el('circle', { cx: x(i), cy: yy, r: 4, fill: css('--panel'), stroke: css('--s3'), 'stroke-width': 1.6 }, s));
      const fx0 = x(Math.floor(n * .66)), fh = y(hi), fl = y(lo + (hi - lo) * .15);
      [0, .236, .382, .5, .618, 1].forEach(k => { const yy = fh + (fl - fh) * k; el('line', { x1: fx0, x2: w - padR, y1: yy, y2: yy, stroke: css('--s4'), opacity: .7, 'stroke-dasharray': k ? '4 3' : '' }, s); el('text', { x: fx0 + 4, y: yy - 3, style: 'fill:' + css('--s4') }, s).textContent = k; });
    }
    host.dataset.last = last.c; host.__data = data; host.__x = x; host.__y = y; host.__pad = padR;
    // crosshair
    const cross = el('g', { style: 'display:none' }, s); const vl = el('line', { class: 'cross', y1: 0, y2: h - padB }, cross), hl = el('line', { class: 'cross', x1: 0, x2: w - padR }, cross);
    const ro = host.querySelector('.ohlc'), tag = el('rect', { width: padR - 4, height: 18, rx: 3, fill: css('--raise'), stroke: css('--line-2') }, cross), tt = el('text', { class: 'lbl' }, cross);
    host.onmousemove = (e) => { const r = host.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top; if (mx > w - padR || my > mainB) { cross.style.display = 'none'; return; }
      cross.style.display = ''; const i = Math.max(0, Math.min(n - 1, Math.round((mx - 8) / ((w - padR - 16) / n) - .5))); vl.setAttribute('x1', x(i)); vl.setAttribute('x2', x(i)); hl.setAttribute('y1', my); hl.setAttribute('y2', my);
      const val = hi + pad - (my - top) / (mainB - top) * (hi + pad - (lo - pad)); tag.setAttribute('x', w - padR + 2); tag.setAttribute('y', my - 9); tt.setAttribute('x', w - padR + 8); tt.setAttribute('y', my + 4); tt.textContent = fmt(val, 0);
      if (ro) { const d = data[i]; ro.querySelector('[data-o]').textContent = fmt(d.o, 0); ro.querySelector('[data-h]').textContent = fmt(d.h, 0); ro.querySelector('[data-l]').textContent = fmt(d.l, 0); ro.querySelector('[data-c]').textContent = fmt(d.c, 0); } };
    host.onmouseleave = () => cross.style.display = 'none';
  }

  function lines(host) {
    const o = JSON.parse(host.dataset.lines); const [s, w, h] = svgFor(host); const padR = o.padR ?? 62, padB = 22, top = 10;
    const series = o.series.map((q, j) => { let d = q.seed ? (() => { const r = rng(q.seed); let v = q.start ?? 100; return Array.from({ length: o.n || 120 }, () => (v *= 1 + (r() - .5) * (q.vol ?? .02) + (q.drift ?? .001))); })() : q.data; if (q.dd) { let pk = d[0]; d = d.map(v => (pk = Math.max(pk, v), (v / pk - 1) * 100 * (q.scale || 1))); } return { ...q, d }; });
    const all = series.flatMap(q => q.d); let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * .08; lo -= pad; hi += pad;
    const n = series[0].d.length, y = scale(hi, lo, top, h - padB), x = (i) => 8 + i * (w - padR - 16) / (n - 1);
    const dec = (hi - lo) < 12 && !o.pct ? 1 : 0; axisY(s, w, y, lo, hi, 4, padR, v => o.pct ? fmt(v, 0) + '%' : fmt(v, dec)); xlabels(s, x, n, h, o.xl || 6, o.labels || MONTHS);
    series.forEach((q, j) => { const col = css('--s' + (q.c || j + 1)); const pts = q.d.map((v, i) => `${x(i)},${y(v)}`).join(' ');
      if (q.area) { const gid = 'g' + Math.random().toString(36).slice(2, 7); const defs = el('defs', {}, s); const g = el('linearGradient', { id: gid, x1: 0, x2: 0, y1: 0, y2: 1 }, defs); el('stop', { offset: 0, 'stop-color': col, 'stop-opacity': .28 }, g); el('stop', { offset: 1, 'stop-color': col, 'stop-opacity': 0 }, g); const base = q.dd ? y(0) : h - padB; el('polygon', { points: `${x(0)},${base} ${pts} ${x(n - 1)},${base}`, fill: `url(#${gid})` }, s); }
      el('polyline', { points: pts, fill: 'none', stroke: col, 'stroke-width': q.w || 1.8, 'stroke-dasharray': q.dash || '', 'stroke-linejoin': 'round', opacity: q.op ?? 1 }, s); });
    if (o.under) { const q = series[0].d; let pk = q[0]; const dd = q.map(v => (pk = Math.max(pk, v), v / pk - 1)); const yy = scale(-.3, 0, h - padB - 60, h - padB); el('polygon', { points: `${x(0)},${yy(0)} ` + dd.map((v, i) => `${x(i)},${yy(v)}`).join(' ') + ` ${x(n - 1)},${yy(0)}`, fill: css('--dn'), opacity: .22 }, s); }
    host.__series = series;
  }

  function bars(host) {
    const o = JSON.parse(host.dataset.bars); const [s, w, h] = svgFor(host); const padB = 22, n = o.n || 40, r = rng(o.seed || 3), padR = 8;
    const vals = Array.from({ length: n }, (_, i) => Math.exp(-Math.pow((i - n * (o.peak ?? .42)) / (n * .2), 2)) * (60 + r() * 30) + r() * 6); const m = Math.max(...vals); const bw = (w - 16) / n;
    const mark = o.mark ?? .3; vals.forEach((v, i) => { const hh = v / m * (h - padB - 14); el('rect', { x: 8 + i * bw + 1, y: h - padB - hh, width: bw - 2, height: hh, rx: 1.5, fill: i / n < (o.zero ?? .13) ? css('--dn') : css('--up'), opacity: .75 }, s); });
    const mx = 8 + mark * (w - 16); el('line', { x1: mx, x2: mx, y1: 6, y2: h - padB, stroke: css('--text'), 'stroke-dasharray': '3 3' }, s); el('text', { x: mx + 6, y: 16, class: 'lbl' }, s).textContent = o.label || 'actual run';
    (o.ticks || ['-40%', '-20%', '0', '+20%', '+40%', '+60%']).forEach((t, i, a) => el('text', { x: 8 + (i + .5) / a.length * (w - 16), y: h - 6, 'text-anchor': 'middle' }, s).textContent = t);
  }

  function donut(host) {
    const parts = JSON.parse(host.dataset.donut); const sz = 150, r = 60, c = 2 * Math.PI * r; host.innerHTML = '';
    const s = el('svg', { viewBox: '0 0 150 150', width: sz, height: sz }); host.append(s); let off = 0; const tot = parts.reduce((a, p) => a + p[1], 0);
    parts.forEach((p, i) => { const len = p[1] / tot * c; el('circle', { cx: 75, cy: 75, r, fill: 'none', stroke: css('--s' + (i + 1)), 'stroke-width': 18, 'stroke-dasharray': `${len - 2} ${c - len + 2}`, 'stroke-dashoffset': -off, transform: 'rotate(-90 75 75)' }, s); off += len; });
  }

  function heat(host) {
    const years = JSON.parse(host.dataset.heat), r = rng(21); const mo = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];
    let h = `<table class="data tight" style="width:auto"><thead><tr><th></th>${mo.map(m => `<th class="r" style="min-width:46px">${m}</th>`).join('')}<th class="r">Year</th></tr></thead><tbody>`;
    years.forEach(y => { let tot = 1; const cells = mo.map(() => { const v = (r() - .42) * 14; tot *= 1 + v / 100; const a = Math.min(.55, Math.abs(v) / 14 * .6); const c = v >= 0 ? 'var(--up)' : 'var(--dn)';
      return `<td class="r" style="background:color-mix(in srgb, ${c} ${Math.round(a * 100)}%, transparent)">${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}</td>`; }).join(''); h += `<tr><td class="muted">${y}</td>${cells}<td class="r ${tot >= 1 ? 'up' : 'dn'}">${((tot - 1) * 100).toFixed(1)}%</td></tr>`; });
    host.innerHTML = h + '</tbody></table>';
  }

  function redraw() {
    document.querySelectorAll('[data-candles]').forEach(candles); document.querySelectorAll('[data-lines]').forEach(lines);
    document.querySelectorAll('[data-bars]').forEach(bars); document.querySelectorAll('[data-donut]').forEach(donut); document.querySelectorAll('[data-heat]').forEach(heat);
  }
  let t; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(redraw, 60); });
  window.TB = { ico, redraw, walk, rng, fmt, fillIcons };
  document.addEventListener('DOMContentLoaded', () => { build(); });
})();
