"use strict";

// ---- Constants --------------------------------------------------------------

const INTERVAL_SECONDS = 5 * 60;
const OB_EXTEND_BARS = 24;
const FVG_EXTEND_BARS = 16;

// Detector chip/render colors (2026-09 rebuild Phase 6). Purple (order block)
// and blue (FVG) carried over unchanged from design-reference/'s mockup; orange
// carried from that mockup's retired "liq sweep" swatch to fakeout/trap; teal
// and rose are new, chosen to avoid red/amber (reserved for mode meaning
// elsewhere - kill switch, live danger states). BOS/CHoCH's bos/choch colors
// and the swing dot colors are retired along with their overlay (see
// docs/detector-logic.md's BOS/CHoCH section for why the detector itself still
// runs - its swing-confirmation machinery feeds trendline/channel fitting -
// even though nothing draws it on the chart anymore).
const COLORS = {
  obBullish: "rgba(62,207,142,0.8)",
  obBullishFill: "rgba(62,207,142,0.20)",
  obBearish: "rgba(224,72,61,0.8)",
  obBearishFill: "rgba(224,72,61,0.20)",
  fvgBullish: "rgba(56,189,248,0.8)",
  fvgBullishFill: "rgba(56,189,248,0.16)",
  fvgBearish: "rgba(217,154,43,0.8)",
  fvgBearishFill: "rgba(217,154,43,0.16)",
  trendline: "#2dd4bf",
  channelStroke: "rgba(244,114,182,0.75)",
  channelFill: "rgba(244,114,182,0.12)",
  fakeoutTrap: "#e07b53",
};

// ---- State --------------------------------------------------------------------

const state = {
  mode: "backtest",
  chart: null,
  series: null,
  canvas: null,
  ctx: null,
  candles: [],
  features: {},
  activeLayers: new Set(["orderBlocks", "fvgs", "trendline", "channel", "fakeoutTrap"]),
  ws: null,
  currentRunId: null,
  trades: [],
  selectedTradeIdx: null,
};

// ---- Helpers ------------------------------------------------------------------

function isoToUnix(iso) {
  return Math.floor(new Date(iso).getTime() / 1000);
}

function fmtUsd(n) {
  return "$" + Number(n).toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}

function fmtPct(n) {
  return (n >= 0 ? "+" : "") + Number(n).toFixed(2) + "%";
}

async function api(path, opts) {
  const resp = await fetch(path, opts);
  return resp.json();
}

let errorBannerTimer = null;

function showError(message) {
  const banner = document.getElementById("errorBanner");
  banner.textContent = message;
  banner.classList.remove("hidden");
  clearTimeout(errorBannerTimer);
  errorBannerTimer = setTimeout(() => banner.classList.add("hidden"), 8000);
}

// ---- Tabs -----------------------------------------------------------------------

function setupTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      document.getElementById("panelBacktest").classList.toggle("hidden", state.mode !== "backtest");
      document.getElementById("panelPaper").classList.toggle("hidden", state.mode !== "paper");
      document.getElementById("panelLive").classList.toggle("hidden", state.mode !== "live");
      if (state.mode === "live") renderRiskSummary();
    });
  });
}

// ---- Chart ----------------------------------------------------------------------

function initChart() {
  const container = document.getElementById("chart");
  state.chart = LightweightCharts.createChart(container, {
    layout: { background: { color: "#101216" }, textColor: "#79828c", fontFamily: "IBM Plex Mono, monospace" },
    grid: { vertLines: { color: "#1c2026" }, horzLines: { color: "#1c2026" } },
    rightPriceScale: { borderColor: "#23272e" },
    timeScale: { borderColor: "#23272e", timeVisible: true, secondsVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    autoSize: true,
  });
  state.series = state.chart.addCandlestickSeries({
    upColor: "#e8eaed", downColor: "#6b7078", borderVisible: false,
    wickUpColor: "#e8eaed", wickDownColor: "#6b7078",
  });

  state.canvas = document.getElementById("overlayCanvas");
  state.ctx = state.canvas.getContext("2d");

  const redraw = () => drawOverlays();
  state.chart.timeScale().subscribeVisibleTimeRangeChange(redraw);
  new ResizeObserver(() => {
    resizeCanvas();
    drawOverlays();
  }).observe(container);
}

function resizeCanvas() {
  const container = document.getElementById("chart");
  const rect = container.getBoundingClientRect();
  state.canvas.width = rect.width;
  state.canvas.height = rect.height;
}

async function loadSnapshot(filename) {
  const data = await api(`/api/snapshot/${encodeURIComponent(filename)}`);
  if (data.error) {
    console.error(data.error);
    return;
  }
  state.candles = data.candles;
  state.features = data.features || {};
  state.series.setData(data.candles);
  state.chart.timeScale().fitContent();
  if (data.candles.length) {
    document.getElementById("lastPrice").textContent = fmtUsd(data.candles[data.candles.length - 1].close);
  }
  resizeCanvas();
  drawOverlays();
}

function drawOverlays() {
  if (!state.ctx) return;
  const ctx = state.ctx;
  const w = state.canvas.width, h = state.canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!state.candles.length) return;

  const timeScale = state.chart.timeScale();
  const rightEdgeX = w;

  if (state.activeLayers.has("orderBlocks")) {
    for (const ob of state.features.order_blocks || []) {
      drawBox(ctx, timeScale, isoToUnix(ob.time), ob.time && isoToUnix(ob.time) + OB_EXTEND_BARS * INTERVAL_SECONDS,
        ob.low, ob.high, rightEdgeX,
        ob.type === "bullish" ? COLORS.obBullish : COLORS.obBearish,
        ob.type === "bullish" ? COLORS.obBullishFill : COLORS.obBearishFill);
    }
  }

  if (state.activeLayers.has("fvgs")) {
    for (const fvg of state.features.fvgs || []) {
      drawBox(ctx, timeScale, isoToUnix(fvg.time), isoToUnix(fvg.time) + FVG_EXTEND_BARS * INTERVAL_SECONDS,
        fvg.bottom, fvg.top, rightEdgeX,
        fvg.type === "bullish" ? COLORS.fvgBullish : COLORS.fvgBearish,
        fvg.type === "bullish" ? COLORS.fvgBullishFill : COLORS.fvgBearishFill);
    }
  }

  if (state.activeLayers.has("trendline")) {
    const trendline = state.features.trendline || {};
    if (trendline.support) drawTrendlineLine(ctx, timeScale, trendline.support, rightEdgeX);
    if (trendline.resistance) drawTrendlineLine(ctx, timeScale, trendline.resistance, rightEdgeX);
    for (const brk of trendline.breaks || []) {
      drawPointMarker(ctx, timeScale, brk.time, brk.price, COLORS.trendline, brk.type);
    }
  }

  if (state.activeLayers.has("channel")) {
    const channel = state.features.channel || {};
    if (channel.ascending) drawChannelBand(ctx, timeScale, channel.ascending, rightEdgeX);
    if (channel.descending) drawChannelBand(ctx, timeScale, channel.descending, rightEdgeX);
  }

  if (state.activeLayers.has("fakeoutTrap")) {
    const fk = (state.features.fakeout || {}).fakeout;
    const trap = (state.features.fakeout || {}).trap;
    if (fk) drawPointMarker(ctx, timeScale, fk.time, fk.swept_level, COLORS.fakeoutTrap, fk.type);
    if (trap) drawPointMarker(ctx, timeScale, trap.time, trap.broken_level, COLORS.fakeoutTrap, trap.type);
  }
}

// detect_trendline/detect_channel fit "index" against the tail lookback_bars
// window compute_smc_features actually passed them (window = df.tail(lookback_bars)
// in data/smc.py), NOT against the full candle array this dashboard loads for
// charting - a loaded snapshot can be millions of rows while the fit only ever
// saw the last ~90. slope*i+intercept is only meaningful for i in
// [0, lookbackWindowCandles().length), mapped back to real chart time via that
// same tail slice's own candle array, not state.candles directly.
function lookbackWindowCandles() {
  const n = state.features.lookback_bars;
  if (!n || n >= state.candles.length) return state.candles;
  return state.candles.slice(-n);
}

// A trendline (support or resistance) is price = slope * windowIndex + intercept -
// drawn as one thin diagonal line spanning the lookback window that produced it.
function drawTrendlineLine(ctx, timeScale, line, rightEdgeX) {
  const window = lookbackWindowCandles();
  const n = window.length;
  if (n < 2) return;
  const x0 = timeScale.timeToCoordinate(window[0].time);
  const x1raw = timeScale.timeToCoordinate(window[n - 1].time);
  const x1 = x1raw === null ? rightEdgeX : x1raw;
  const y0 = state.series.priceToCoordinate(line.slope * 0 + line.intercept);
  const y1 = state.series.priceToCoordinate(line.slope * (n - 1) + line.intercept);
  if (x0 === null || y0 === null || y1 === null) return;
  ctx.save();
  ctx.strokeStyle = COLORS.trendline;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.restore();
}

// Two parallel lines (same slope, lower/upper intercepts) with a low-opacity
// shaded region between them - same "boundary + shaded fill" idiom as the
// existing FVG box rendering, just diagonal instead of axis-aligned.
function drawChannelBand(ctx, timeScale, channel, rightEdgeX) {
  const window = lookbackWindowCandles();
  const n = window.length;
  if (n < 2) return;
  const x0 = timeScale.timeToCoordinate(window[0].time);
  const x1raw = timeScale.timeToCoordinate(window[n - 1].time);
  const x1 = x1raw === null ? rightEdgeX : x1raw;
  const lowerAt = (i) => channel.slope * i + channel.lower.intercept;
  const upperAt = (i) => channel.slope * i + channel.upper.intercept;
  const y0Lower = state.series.priceToCoordinate(lowerAt(0));
  const y1Lower = state.series.priceToCoordinate(lowerAt(n - 1));
  const y0Upper = state.series.priceToCoordinate(upperAt(0));
  const y1Upper = state.series.priceToCoordinate(upperAt(n - 1));
  if ([x0, y0Lower, y1Lower, y0Upper, y1Upper].some((v) => v === null)) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x0, y0Upper);
  ctx.lineTo(x1, y1Upper);
  ctx.lineTo(x1, y1Lower);
  ctx.lineTo(x0, y0Lower);
  ctx.closePath();
  ctx.fillStyle = COLORS.channelFill;
  ctx.fill();
  ctx.strokeStyle = COLORS.channelStroke;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, y0Upper); ctx.lineTo(x1, y1Upper); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x0, y0Lower); ctx.lineTo(x1, y1Lower); ctx.stroke();
  ctx.restore();
}

// Shared point-marker renderer for trendline breaks and fakeout/trap events -
// a small filled diamond plus a text label, at one (time, price) point.
function drawPointMarker(ctx, timeScale, iso, price, color, label) {
  if (!iso) return;
  const x = timeScale.timeToCoordinate(isoToUnix(iso));
  const y = state.series.priceToCoordinate(price);
  if (x === null || y === null) return;
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x, y - 4); ctx.lineTo(x + 4, y); ctx.lineTo(x, y + 4); ctx.lineTo(x - 4, y);
  ctx.closePath();
  ctx.fill();
  ctx.font = "10px IBM Plex Mono, monospace";
  ctx.fillText(label, x + 6, y - 4);
  ctx.restore();
}

function drawBox(ctx, timeScale, t0, t1, priceLow, priceHigh, rightEdgeX, strokeColor, fillColor) {
  const x0 = timeScale.timeToCoordinate(t0);
  if (x0 === null) return;
  const x1raw = t1 ? timeScale.timeToCoordinate(t1) : null;
  const x1 = x1raw === null ? rightEdgeX : x1raw;
  const y0 = state.series.priceToCoordinate(priceHigh);
  const y1 = state.series.priceToCoordinate(priceLow);
  if (y0 === null || y1 === null) return;

  ctx.fillStyle = fillColor;
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
}

function setupOverlayToggles() {
  document.querySelectorAll(".overlay-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const layer = btn.dataset.layer;
      if (state.activeLayers.has(layer)) {
        state.activeLayers.delete(layer);
        btn.classList.remove("active");
      } else {
        state.activeLayers.add(layer);
        btn.classList.add("active");
      }
      drawOverlays();
    });
  });
}

// ---- Snapshots --------------------------------------------------------------------

async function loadSnapshotList() {
  const snapshots = await api("/api/snapshots");
  const select = document.getElementById("snapshotSelect");
  select.innerHTML = "";
  for (const s of snapshots) {
    const opt = document.createElement("option");
    opt.value = s.filename;
    opt.textContent = s.filename;
    select.appendChild(opt);
  }
  if (snapshots.length) {
    await loadSnapshot(snapshots[0].filename);
  }
  select.addEventListener("change", () => loadSnapshot(select.value));
}

// ---- Backtest -----------------------------------------------------------------------

async function runBacktest() {
  const snapshot = document.getElementById("snapshotSelect").value;
  if (!snapshot) return;
  const body = {
    snapshot,
    min_confluence: parseInt(document.getElementById("paramConfluence").value, 10),
    initial_balance: parseFloat(document.getElementById("paramBalance").value),
    taker_fee_pct: parseFloat(document.getElementById("paramFee").value) / 100,
    slippage_bps: parseFloat(document.getElementById("paramSlippage").value),
  };
  const runBtn = document.getElementById("runBacktestBtn");
  runBtn.disabled = true;
  runBtn.textContent = "RUNNING...";
  document.getElementById("progressBox").classList.remove("hidden");
  document.getElementById("resultsRow").classList.add("hidden");

  const result = await api("/api/backtest/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (result.error) {
    showError(result.error);
    runBtn.disabled = false;
    runBtn.textContent = "RUN BACKTEST";
    return;
  }
  state.currentRunId = result.run_id;
}

function onBacktestProgress(msg) {
  if (msg.run_id !== state.currentRunId) return;
  const pct = msg.candles_total ? Math.round((msg.candles_processed / msg.candles_total) * 100) : 0;
  document.getElementById("progressPct").textContent = pct + "%";
  document.getElementById("progressBar").style.width = pct + "%";
  document.getElementById("progressCandles").textContent = `${msg.candles_processed} / ${msg.candles_total} candles`;
  document.getElementById("statusBacktest").textContent = "running";
}

function onBacktestDone(msg) {
  if (msg.run_id !== state.currentRunId) return;
  document.getElementById("progressBox").classList.add("hidden");
  const runBtn = document.getElementById("runBacktestBtn");
  runBtn.disabled = false;
  runBtn.textContent = "RUN BACKTEST";
  document.getElementById("statusBacktest").textContent = "done";
  renderResults(msg.result);
}

function onBacktestError(msg) {
  if (msg.run_id !== state.currentRunId) return;
  document.getElementById("progressBox").classList.add("hidden");
  const runBtn = document.getElementById("runBacktestBtn");
  runBtn.disabled = false;
  runBtn.textContent = "RUN BACKTEST";
  document.getElementById("statusBacktest").textContent = "error";
  showError("Backtest failed: " + msg.message);
}

function renderResults(result) {
  document.getElementById("resultsRow").classList.remove("hidden");
  renderMetrics(result.metrics);
  renderEquityCurve(result.equity_curve, result.buy_hold_curve);
  renderTradeLog(result.trades);
}

function renderMetrics(m) {
  const rows = [
    ["Total trades", m.total_trades, "neutral"],
    ["Win rate", m.win_rate_pct.toFixed(1) + "%", m.win_rate_pct >= 50 ? "pos" : "neutral"],
    ["Profit factor", isFinite(m.profit_factor) ? m.profit_factor.toFixed(2) : "∞", m.profit_factor >= 1 ? "pos" : "neg"],
    ["Total PnL", fmtUsd(m.total_pnl), m.total_pnl >= 0 ? "pos" : "neg"],
    ["ROI", fmtPct(m.roi_pct), m.roi_pct >= 0 ? "pos" : "neg"],
    ["Max drawdown", m.max_drawdown_pct.toFixed(2) + "%", "neg"],
    ["Avg R multiple", m.avg_r_multiple.toFixed(2) + "R", m.avg_r_multiple >= 0 ? "pos" : "neg"],
    ["Buy & hold", m.buy_hold_return_pct === null ? "--" : fmtPct(m.buy_hold_return_pct), "neutral"],
  ];
  const container = document.getElementById("metricsList");
  container.innerHTML = "";
  for (const [label, value, cls] of rows) {
    const row = document.createElement("div");
    row.className = "metric-row";
    row.innerHTML = `<span class="metric-label">${label}</span><span class="metric-value ${cls}">${value}</span>`;
    container.appendChild(row);
  }
}

function renderEquityCurve(equity, buyHold) {
  const svg = document.getElementById("equitySvg");
  svg.innerHTML = "";
  const W = 600, H = 200;

  const grid = [199, 150, 100, 50];
  for (const y of grid) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", 0); line.setAttribute("y1", y);
    line.setAttribute("x2", W); line.setAttribute("y2", y);
    line.setAttribute("stroke", "#1c2026"); line.setAttribute("stroke-width", "1");
    svg.appendChild(line);
  }

  const allBalances = [...equity, ...buyHold].map((p) => p.balance);
  const min = Math.min(...allBalances), max = Math.max(...allBalances);
  const range = max - min || 1;

  const toPoints = (series) =>
    series.map((p, i) => {
      const x = (i / Math.max(1, series.length - 1)) * W;
      const y = H - ((p.balance - min) / range) * H;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");

  if (buyHold.length) {
    const bh = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    bh.setAttribute("points", toPoints(buyHold));
    bh.setAttribute("fill", "none"); bh.setAttribute("stroke", "#565d66");
    bh.setAttribute("stroke-width", "1.5"); bh.setAttribute("stroke-dasharray", "4 3");
    svg.appendChild(bh);
  }
  if (equity.length) {
    const eq = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    eq.setAttribute("points", toPoints(equity));
    eq.setAttribute("fill", "none"); eq.setAttribute("stroke", "#38bdf8"); eq.setAttribute("stroke-width", "1.75");
    svg.appendChild(eq);
  }
}

function renderTradeLog(trades) {
  state.trades = trades;
  state.selectedTradeIdx = trades.length ? trades.length - 1 : null; // most recent, by default

  const list = document.getElementById("feedList");
  document.getElementById("feedMeta").textContent = `${trades.length} trades`;
  if (!trades.length) {
    list.innerHTML = '<div class="feed-empty">No trades in this run.</div>';
    renderBreakdown(null);
    return;
  }
  list.innerHTML = "";
  for (let i = trades.length - 1; i >= 0; i--) {
    const t = trades[i];
    const row = document.createElement("div");
    row.className = "feed-row" + (i === state.selectedTradeIdx ? " selected" : "");
    row.dataset.idx = i;
    const sideClass = t.side === "BUY" ? "buy" : "sell";
    const pnlClass = t.pnl >= 0 ? "pos" : "neg";
    row.innerHTML = `
      <span class="feed-bar ${sideClass}"></span>
      <div class="feed-main">
        <div class="feed-top">
          <span class="feed-side ${sideClass}">${t.side}</span>
          <span class="feed-price">${t.entry_price.toFixed(1)} → ${t.exit_price.toFixed(1)}</span>
        </div>
        <span class="feed-sub">${t.entry_time.slice(0, 16).replace("T", " ")} · ${t.exit_reason} · conf ${t.confluence}</span>
      </div>
      <div class="feed-pnl">
        <span class="feed-pnl-val ${pnlClass}">${fmtUsd(t.pnl)}</span>
        <span class="feed-r">${t.r_multiple.toFixed(2)}R</span>
      </div>`;
    row.addEventListener("click", () => selectTrade(i));
    list.appendChild(row);
  }
  renderBreakdown(trades[state.selectedTradeIdx]);
}

function selectTrade(idx) {
  state.selectedTradeIdx = idx;
  for (const row of document.querySelectorAll("#feedList .feed-row")) {
    row.classList.toggle("selected", Number(row.dataset.idx) === idx);
  }
  renderBreakdown(state.trades[idx]);
}

// Detector breakdown panel (2026-09 rebuild Phase 6) - the 5-vote breakdown for
// whichever trade is selected in the log above (most recent by default), read
// straight from SMCSignal.contributing_factors as carried onto Trade
// (backtest/runner.py) and serialized by viz/control.py.
const DETECTOR_LABELS = [
  ["order_block", "Order Block", "ob"],
  ["fvg", "FVG", "fvg"],
  ["trendline", "Trendline", "trendline"],
  ["channel", "Channel", "channel"],
  ["fakeout_trap", "Fakeout/Trap", "fakeout-trap"],
];

function renderBreakdown(trade) {
  const list = document.getElementById("breakdownList");
  const meta = document.getElementById("breakdownMeta");
  if (!trade) {
    meta.textContent = "--";
    list.innerHTML = '<div class="feed-empty">Click a trade in the log above to see its 5-vote breakdown.</div>';
    return;
  }
  const factors = trade.contributing_factors || {};
  meta.textContent = `conf ${trade.confluence >= 0 ? "+" : ""}${trade.confluence}`;
  list.innerHTML = "";
  for (const [key, label, swatchClass] of DETECTOR_LABELS) {
    const vote = factors[key] ?? null;
    const voteClass = vote === "bullish" ? "bullish" : vote === "bearish" ? "bearish" : "none";
    const voteText = vote === "bullish" ? "BULLISH" : vote === "bearish" ? "BEARISH" : "—";
    const row = document.createElement("div");
    row.className = "breakdown-row";
    row.innerHTML = `
      <span class="breakdown-swatch swatch ${swatchClass}"></span>
      <span class="breakdown-name">${label}</span>
      <span class="breakdown-vote ${voteClass}">${voteText}</span>`;
    list.appendChild(row);
  }
}

// ---- Paper / Live -----------------------------------------------------------------

const CONNECTION_COLORS = { healthy: "#3ecf8e", reconnecting: "#d99a2b", dropped: "#565d66" };

async function setupPaper() {
  document.getElementById("paperToggleBtn").addEventListener("click", async () => {
    const status = await api("/api/paper/status");
    if (status.status === "running") {
      await api("/api/paper/stop", { method: "POST" });
    } else {
      const result = await api("/api/paper/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (result.error) showError(result.error);
    }
    refreshPaperStatus();
  });
  refreshPaperStatus();
}

async function refreshPaperStatus() {
  const s = await api("/api/paper/status");
  document.getElementById("statusPaper").textContent = s.status;
  document.getElementById("paperBalance").textContent = s.balance !== null ? fmtUsd(s.balance) : "$--";
  document.getElementById("paperPosition").textContent = s.open_position
    ? `${s.open_position.side} ${s.open_position.size.toFixed(5)} @ ${s.open_position.entry_price.toFixed(1)}`
    : "flat";
  document.getElementById("connLabel").textContent = s.connection;
  document.getElementById("connDot").style.background = CONNECTION_COLORS[s.connection] || "#565d66";
  document.getElementById("dotPaper").style.background = s.status === "running" ? "#d99a2b" : "#565d66";

  const btn = document.getElementById("paperToggleBtn");
  btn.textContent = s.status === "running" ? "STOP PAPER TRADING" : "START PAPER TRADING";
}

async function setupLive() {
  const readiness = await api("/api/live/readiness");
  document.getElementById("killState").textContent = readiness.kill_switch_ready ? "READY" : "NOT READY";
  document.getElementById("killReason").textContent = readiness.reason;
  const btn = document.getElementById("liveStartBtn");
  btn.disabled = !readiness.kill_switch_ready;
  document.getElementById("statusLive").textContent = readiness.kill_switch_ready ? "ready" : "disabled";
  document.getElementById("dotLive").classList.toggle("red", !readiness.kill_switch_ready);
  await renderRiskSummary();
}

async function renderRiskSummary() {
  const s = await api("/api/live/risk-summary");
  const rows = [
    ["POSITION SIZE / TRADE", `${s.position_size_pct.toFixed(1)}% of balance`, false],
    ["DAILY LOSS LIMIT", s.daily_loss_limit === null ? "not configured" : fmtUsd(s.daily_loss_limit), s.daily_loss_limit === null],
    ["MAX DRAWDOWN AUTO-HALT", s.max_drawdown_halt_pct === null ? "not configured" : s.max_drawdown_halt_pct.toFixed(1) + "%", s.max_drawdown_halt_pct === null],
    ["CONFLUENCE THRESHOLD", s.confluence_threshold_display, false],
    ["BALANCE AT ARM", s.balance_at_arm === null ? "--" : fmtUsd(s.balance_at_arm), s.balance_at_arm === null],
  ];
  const list = document.getElementById("riskSummaryList");
  list.innerHTML = "";
  for (const [label, value, dim] of rows) {
    const row = document.createElement("div");
    row.className = "risk-row";
    row.innerHTML = `<span class="risk-label">${label}</span><span class="risk-value${dim ? " dim" : ""}">${value}</span>`;
    list.appendChild(row);
  }
}

// ---- WebSocket ----------------------------------------------------------------------

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
      case "backtest_progress": onBacktestProgress(msg); break;
      case "backtest_done": onBacktestDone(msg); break;
      case "backtest_error": onBacktestError(msg); break;
      case "paper_status": refreshPaperStatus(); break;
      case "paper_trade": console.log("[paper trade]", msg); break;
      default: break;
    }
  };
  ws.onclose = () => {
    setTimeout(connectWs, 2000); // dashboard-socket reconnect only; not Kraken's own feed
  };
}

// ---- Init -----------------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  initChart();
  setupOverlayToggles();
  document.getElementById("runBacktestBtn").addEventListener("click", runBacktest);
  await loadSnapshotList();
  await setupPaper();
  await setupLive();
  connectWs();
});
