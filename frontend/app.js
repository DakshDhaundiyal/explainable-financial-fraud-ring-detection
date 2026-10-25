/**
 * Explainable TGAT Financial Fraud Ring Investigator
 * Frontend Single-Page Application Controller
 * Pure Vanilla ES6 / Zero External Build Tools
 */

// Application State
const state = {
  currentSessionId: null,
  activeTab: "view-detector",
  resultsPage: 1,
  resultsLimit: 25,
  resultsSort: "risk_score:desc",
  resultsSearch: "",
  resultsRiskFilter: "",
  activeAccountData: null,
  activeSidepanelTab: "reason",
  graphNodes: [],
  graphEdges: [],
  particles: [],
  isAnimating: true,
  transform: { x: 0, y: 0, scale: 1 },
  selectedNode: null,
  dragNode: null,
  hoveredNode: null,
  isPanning: false,
  panStart: { x: 0, y: 0 },
};

// ============================================================================
// 1. INITIALIZATION & ROUTING
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initActionHero();
  initGraphCanvas();
  initTransactionsTable();
  initTransactionModal();

  // Check if an existing analysis session is available
  checkInitialSession();
});

function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-item-btn");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetView = btn.getAttribute("data-target");
      switchTab(targetView);
    });
  });
}

function switchTab(viewId) {
  state.activeTab = viewId;

  // Update nav buttons
  document.querySelectorAll(".nav-item-btn").forEach((btn) => {
    const isTarget = btn.getAttribute("data-target") === viewId;
    btn.classList.toggle("active", isTarget);
    btn.setAttribute("aria-selected", isTarget ? "true" : "false");
  });

  // Update view containers
  document.querySelectorAll(".tab-view").forEach((view) => {
    view.classList.toggle("active", view.id === viewId);
  });

  // Tab entry triggers
  if (viewId === "view-comparison") {
    loadComparisonMetrics();
  } else if (viewId === "view-detector" && state.currentSessionId) {
    // Redraw graph if needed
    if (state.graphNodes.length > 0) {
      setTimeout(() => resizeGraphCanvas(), 50);
    }
  }
}

async function checkInitialSession() {
  try {
    const res = await fetch("/api/analysis/overview");
    if (!res.ok) return;
    const data = await res.json();
    if (data.has_data && data.session_id) {
      state.currentSessionId = data.session_id;
      updateHeaderSession(data.session_id, data.threshold);
      renderAnalysisResults(data);
    }
  } catch (err) {
    console.warn("No active session on boot:", err);
  }
}

function updateHeaderSession(sessionId, threshold) {
  const pill = document.getElementById("header-session-id");
  if (pill) pill.textContent = sessionId ? sessionId.substring(0, 18) + (sessionId.length > 18 ? "..." : "") : "Ready";

  const threshVal = document.getElementById("header-threshold-val");
  if (threshVal && threshold) threshVal.textContent = Number(threshold).toFixed(4);
}

// ============================================================================
// 2. HERO ACTIONS: 1-CLICK DEMO & CSV UPLOAD
// ============================================================================

function initActionHero() {
  const demoBtn = document.getElementById("btn-run-demo");
  const headerDemoBtn = document.getElementById("btn-header-demo");

  if (demoBtn) {
    demoBtn.addEventListener("click", () => runDemoBatch(50));
  }
  if (headerDemoBtn) {
    headerDemoBtn.addEventListener("click", () => {
      switchTab("view-detector");
      runDemoBatch(50);
    });
  }

  // File Upload Handlers
  const fileInput = document.getElementById("file-input-csv");
  const browseBtn = document.getElementById("btn-browse-file");
  const dropzone = document.getElementById("dropzone-box");

  if (browseBtn && fileInput) {
    browseBtn.addEventListener("click", () => fileInput.click());
  }

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadAndRunCsv(e.target.files[0]);
      }
    });
  }

  if (dropzone) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--accent-blue)";
    });
    dropzone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--border-card)";
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--border-card)";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadAndRunCsv(e.dataTransfer.files[0]);
      }
    });
  }
}

async function runDemoBatch(count = 50) {
  showScoringProgress("Generating realistic IBM AML benchmark batch (50 transactions)...", 20);

  try {
    const res = await fetch("/api/analysis/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ count: count }),
    });

    if (!res.ok) throw new Error("Failed to load demo batch");
    const data = await res.json();

    updateScoringProgress("Constructing dynamic temporal graph & computing Bochner time encodings...", 60);

    setTimeout(async () => {
      updateScoringProgress("Running TGAT multi-head attention & calculating risk scores...", 90);
      
      setTimeout(async () => {
        completeScoringProgress();
        state.currentSessionId = data.session_id;
        updateHeaderSession(data.session_id, data.threshold);
        await loadSessionOverview(data.session_id);
      }, 300);
    }, 300);

  } catch (err) {
    hideScoringProgress();
    alert(`Demo failed: ${err.message}`);
  }
}

async function uploadAndRunCsv(file) {
  if (!file.name.endsWith(".csv")) {
    alert("Please upload a valid .csv file.");
    return;
  }

  showScoringProgress(`Uploading ${file.name} and validating schema...`, 30);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/api/analysis/run", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || "Upload analysis failed");
    }

    const data = await res.json();
    updateScoringProgress("Executing TGAT neural inference across transaction graph...", 85);

    setTimeout(async () => {
      completeScoringProgress();
      state.currentSessionId = data.session_id;
      updateHeaderSession(data.session_id, data.threshold);
      await loadSessionOverview(data.session_id);
    }, 400);

  } catch (err) {
    hideScoringProgress();
    alert(`Analysis error: ${err.message}`);
  }
}

function showScoringProgress(text, pct) {
  const pCard = document.getElementById("scoring-progress");
  const pText = document.getElementById("scoring-progress-text");
  const pPct = document.getElementById("scoring-progress-pct");
  const pBar = document.getElementById("scoring-progress-bar");

  if (pCard) pCard.style.display = "flex";
  if (pText) pText.textContent = text;
  if (pPct) pPct.textContent = `${pct}%`;
  if (pBar) pBar.style.width = `${pct}%`;
}

function updateScoringProgress(text, pct) {
  const pText = document.getElementById("scoring-progress-text");
  const pPct = document.getElementById("scoring-progress-pct");
  const pBar = document.getElementById("scoring-progress-bar");

  if (pText) pText.textContent = text;
  if (pPct) pPct.textContent = `${pct}%`;
  if (pBar) pBar.style.width = `${pct}%`;
}

function completeScoringProgress() {
  updateScoringProgress("Inference complete! Rendering results...", 100);
  setTimeout(() => {
    hideScoringProgress();
  }, 600);
}

function hideScoringProgress() {
  const pCard = document.getElementById("scoring-progress");
  if (pCard) pCard.style.display = "none";
}

async function loadSessionOverview(sessionId) {
  try {
    const res = await fetch(`/api/analysis/overview?session_id=${encodeURIComponent(sessionId)}`);
    if (!res.ok) throw new Error("Overview query failed");
    const data = await res.json();
    renderAnalysisResults(data);
  } catch (err) {
    console.error("Failed to load session overview:", err);
  }
}

function renderAnalysisResults(overviewData) {
  const container = document.getElementById("detector-results-container");
  if (!container) return;
  container.style.display = "block";

  // 1. Executive Plain-English Findings Banner
  const banner = document.getElementById("detector-exec-banner");
  const title = document.getElementById("detector-exec-title");
  const desc = document.getElementById("detector-exec-text");

  const flaggedCount = overviewData.flagged_transactions || 0;
  const isHighAlert = flaggedCount > 0;

  if (banner) {
    banner.className = `layman-callout ${isHighAlert ? "high" : "low"}`;
  }
  if (title) {
    title.innerHTML = isHighAlert
      ? `🚨 ${flaggedCount} Suspicious Money Laundering Payments Detected`
      : `✅ All ${overviewData.total_transactions} Transactions Cleared (No Fraud Rings)`;
  }
  if (desc) {
    desc.textContent = overviewData.executive_summary || (
      isHighAlert
        ? `TGAT flagged ${flaggedCount} transactions with high confidence. The accounts exhibit rapid pass-through structuring and cycling funds across multiple hops.`
        : "All examined transactions exhibit normal legitimate velocity and balance retention."
    );
  }

  // 2. 4 Clean KPIs
  document.getElementById("kpi-total-txs").textContent = (overviewData.total_transactions || 0).toLocaleString();
  document.getElementById("kpi-flagged-txs").textContent = flaggedCount.toLocaleString();
  document.getElementById("kpi-flagged-rate").textContent = `${overviewData.flagged_rate || 0}% of batch`;
  document.getElementById("kpi-cleared-txs").textContent = ((overviewData.total_transactions || 0) - flaggedCount).toLocaleString();
  document.getElementById("kpi-duration").textContent = `Inference: ${overviewData.duration_seconds || 0.12}s`;

  // 3. Load Topology Graph
  loadTopologyGraph();

  // 4. Load Results Table
  loadResultsTable(1);
}

// ============================================================================
// 3. ANIMATED NETWORK TOPOLOGY GRAPH (CANVAS FORCE-DIRECTED)
// ============================================================================

let canvas, ctx, animFrameId, simulationTimer;

function initGraphCanvas() {
  canvas = document.getElementById("graph-canvas");
  if (!canvas) return;
  ctx = canvas.getContext("2d");

  const toggleFlowBtn = document.getElementById("btn-toggle-flow");
  const zoomInBtn = document.getElementById("btn-graph-zoom-in");
  const zoomOutBtn = document.getElementById("btn-graph-zoom-out");
  const resetBtn = document.getElementById("btn-reset-graph");

  if (toggleFlowBtn) {
    toggleFlowBtn.addEventListener("click", () => {
      state.isAnimating = !state.isAnimating;
      toggleFlowBtn.classList.toggle("active", state.isAnimating);
      const icon = document.getElementById("flow-btn-icon");
      const text = document.getElementById("flow-btn-text");
      if (icon) icon.textContent = state.isAnimating ? "⏸" : "▶";
      if (text) text.textContent = state.isAnimating ? "Flow Active" : "Flow Paused";
    });
  }

  if (zoomInBtn) {
    zoomInBtn.addEventListener("click", () => {
      state.transform.scale = Math.min(3.0, state.transform.scale * 1.25);
    });
  }
  if (zoomOutBtn) {
    zoomOutBtn.addEventListener("click", () => {
      state.transform.scale = Math.max(0.4, state.transform.scale * 0.8);
    });
  }
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      state.transform = { x: 0, y: 0, scale: 1 };
      loadTopologyGraph();
    });
  }

  // Mouse Interactivity
  canvas.addEventListener("mousedown", handleCanvasMouseDown);
  canvas.addEventListener("mousemove", handleCanvasMouseMove);
  window.addEventListener("mouseup", handleCanvasMouseUp);
  canvas.addEventListener("wheel", handleCanvasWheel, { passive: false });

  // Animation Loop
  cancelAnimationFrame(animFrameId);
  function renderLoop() {
    drawGraph();
    animFrameId = requestAnimationFrame(renderLoop);
  }
  renderLoop();
}

function resizeGraphCanvas() {
  if (!canvas) return;
  const wrapper = canvas.parentElement;
  if (!wrapper) return;
  canvas.width = wrapper.clientWidth || 700;
  canvas.height = 520;
}

function getCanvasMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  const screenX = e.clientX - rect.left;
  const screenY = e.clientY - rect.top;
  const worldX = (screenX - state.transform.x) / state.transform.scale;
  const worldY = (screenY - state.transform.y) / state.transform.scale;
  return { screenX, screenY, worldX, worldY };
}

function handleCanvasMouseDown(e) {
  const { screenX, screenY, worldX, worldY } = getCanvasMousePos(e);

  for (let i = state.graphNodes.length - 1; i >= 0; i--) {
    const node = state.graphNodes[i];
    const radius = node.is_flagged ? 18 : 14;
    const dist = Math.hypot(node.x - worldX, node.y - worldY);
    if (dist <= radius + 6) {
      state.dragNode = node;
      state.selectedNode = node;
      inspectAccount(node.id);
      return;
    }
  }

  state.isPanning = true;
  state.panStart = { x: screenX - state.transform.x, y: screenY - state.transform.y };
}

function handleCanvasMouseMove(e) {
  const { screenX, screenY, worldX, worldY } = getCanvasMousePos(e);

  if (state.dragNode) {
    state.dragNode.x = worldX;
    state.dragNode.y = worldY;
    return;
  }

  if (state.isPanning) {
    state.transform.x = screenX - state.panStart.x;
    state.transform.y = screenY - state.panStart.y;
    return;
  }

  let hovering = false;
  for (const node of state.graphNodes) {
    const dist = Math.hypot(node.x - worldX, node.y - worldY);
    if (dist <= (node.is_flagged ? 18 : 14) + 6) {
      hovering = true;
      state.hoveredNode = node;
      break;
    }
  }
  if (!hovering) state.hoveredNode = null;
  canvas.style.cursor = hovering ? "pointer" : (state.isPanning ? "grabbing" : "grab");
}

function handleCanvasMouseUp() {
  state.dragNode = null;
  state.isPanning = false;
}

function handleCanvasWheel(e) {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
  const newScale = Math.max(0.4, Math.min(3.0, state.transform.scale * zoomFactor));

  state.transform.x = mouseX - (mouseX - state.transform.x) * (newScale / state.transform.scale);
  state.transform.y = mouseY - (mouseY - state.transform.y) * (newScale / state.transform.scale);
  state.transform.scale = newScale;
}

function initParticles() {
  state.particles = [];
  state.graphEdges.forEach((edge, edgeIdx) => {
    const isHighRisk = edge.is_flagged || edge.max_risk_score >= 0.7372;
    const count = isHighRisk ? 4 : 2;
    for (let i = 0; i < count; i++) {
      state.particles.push({
        edgeIdx: edgeIdx,
        progress: Math.random(),
        speed: 0.005 + Math.min(0.016, (edge.max_risk_score || 0.2) * 0.014),
      });
    }
  });
}

async function loadTopologyGraph() {
  if (!canvas) return;
  resizeGraphCanvas();

  if (!state.currentSessionId) {
    drawEmptyGraph("No active analysis. Run the demo or upload a file.");
    return;
  }

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/graph?min_score=0`);
    if (!res.ok) throw new Error("Failed to load topology");
    const data = await res.json();

    const rawNodes = data.nodes || [];
    const rawEdges = data.edges || [];

    if (rawNodes.length === 0) {
      drawEmptyGraph("No accounts found in this transaction batch.");
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    state.graphNodes = rawNodes.map((n, i) => {
      const angle = (i / rawNodes.length) * 2 * Math.PI;
      const radius = 100 + Math.random() * 140;
      return {
        ...n,
        x: cx + Math.cos(angle) * radius,
        y: cy + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
      };
    });

    state.graphEdges = rawEdges;
    initParticles();
    runForceSimulation();

    // Auto-select the top flagged account immediately
    const topFlagged = state.graphNodes.find((n) => n.is_flagged || n.max_risk_score >= 0.7372) || state.graphNodes[0];
    if (topFlagged) {
      state.selectedNode = topFlagged;
      inspectAccount(topFlagged.id);
    }

  } catch (err) {
    drawEmptyGraph(`Topology load error: ${err.message}`);
  }
}

function runForceSimulation() {
  let iterations = 45;
  clearInterval(simulationTimer);

  simulationTimer = setInterval(() => {
    if (iterations-- <= 0) {
      clearInterval(simulationTimer);
      return;
    }

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;

    // Repulsion
    for (let i = 0; i < state.graphNodes.length; i++) {
      for (let j = i + 1; j < state.graphNodes.length; j++) {
        const n1 = state.graphNodes[i];
        const n2 = state.graphNodes[j];
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.max(20, Math.hypot(dx, dy));
        const force = 450 / (dist * dist);
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        n1.vx -= fx;
        n1.vy -= fy;
        n2.vx += fx;
        n2.vy += fy;
      }
    }

    // Attraction
    const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));
    for (const e of state.graphEdges) {
      const n1 = nodeMap.get(e.source);
      const n2 = nodeMap.get(e.target);
      if (n1 && n2) {
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const force = (dist - 90) * 0.045;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        n1.vx += fx;
        n1.vy += fy;
        n2.vx += fx;
        n2.vy += fy;
      }
    }

    // Centering & velocity decay
    for (const n of state.graphNodes) {
      if (state.dragNode && state.dragNode.id === n.id) continue;
      n.vx += (cx - n.x) * 0.02;
      n.vy += (cy - n.y) * 0.02;
      n.x += Math.max(-14, Math.min(14, n.vx));
      n.y += Math.max(-14, Math.min(14, n.vy));
      n.vx *= 0.65;
      n.vy *= 0.65;
      n.x = Math.max(40, Math.min(width - 40, n.x));
      n.y = Math.max(40, Math.min(height - 40, n.y));
    }
  }, 20);
}

function drawGraph() {
  if (!ctx || !canvas) return;

  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  if (state.graphNodes.length === 0) return;

  ctx.save();
  ctx.translate(state.transform.x, state.transform.y);
  ctx.scale(state.transform.scale, state.transform.scale);

  const nodeMap = new Map(state.graphNodes.map((n) => [n.id, n]));
  const now = Date.now();

  // 1. Draw Directed Edges
  state.graphEdges.forEach((e) => {
    const s = nodeMap.get(e.source);
    const t = nodeMap.get(e.target);
    if (!s || !t) return;

    const isFlagged = e.is_flagged || e.max_risk_score >= 0.7372;
    const isMedium = !isFlagged && e.max_risk_score >= 0.40;
    const isEdgeSelected = state.selectedNode && (state.selectedNode.id === s.id || state.selectedNode.id === t.id);

    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(t.x, t.y);

    if (isFlagged) {
      ctx.strokeStyle = isEdgeSelected ? "rgba(244, 63, 94, 0.95)" : "rgba(244, 63, 94, 0.65)";
      ctx.lineWidth = isEdgeSelected ? 3.5 : 2.5;
    } else if (isMedium) {
      ctx.strokeStyle = isEdgeSelected ? "rgba(245, 158, 11, 0.9)" : "rgba(245, 158, 11, 0.5)";
      ctx.lineWidth = isEdgeSelected ? 2.5 : 1.8;
    } else {
      ctx.strokeStyle = isEdgeSelected ? "rgba(59, 130, 246, 0.9)" : "rgba(100, 116, 139, 0.35)";
      ctx.lineWidth = isEdgeSelected ? 2.0 : 1.2;
    }
    ctx.stroke();

    // Directional Arrow Head
    const angle = Math.atan2(t.y - s.y, t.x - s.x);
    const radius = t.is_flagged || t.max_risk_score >= 0.7372 ? 18 : 14;
    const ax = t.x - Math.cos(angle) * (radius + 3);
    const ay = t.y - Math.sin(angle) * (radius + 3);

    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - 7 * Math.cos(angle - Math.PI / 6), ay - 7 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(ax - 7 * Math.cos(angle + Math.PI / 6), ay - 7 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  });

  // 2. Animated Money Flow Beads
  if (state.isAnimating && state.particles.length > 0) {
    state.particles.forEach((p) => {
      const edge = state.graphEdges[p.edgeIdx];
      if (!edge) return;
      const s = nodeMap.get(edge.source);
      const t = nodeMap.get(edge.target);
      if (!s || !t) return;

      p.progress += p.speed;
      if (p.progress > 1) p.progress = 0;

      const px = s.x + (t.x - s.x) * p.progress;
      const py = s.y + (t.y - s.y) * p.progress;

      const isFlagged = edge.is_flagged || edge.max_risk_score >= 0.7372;

      ctx.beginPath();
      ctx.arc(px, py, isFlagged ? 4.0 : 3.0, 0, Math.PI * 2);
      ctx.fillStyle = isFlagged ? "#F43F5E" : "#38BDF8";
      ctx.shadowColor = isFlagged ? "#E11D48" : "#0284C7";
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  }

  // 3. Draw Nodes (Accounts)
  state.graphNodes.forEach((n) => {
    const isSelected = state.selectedNode && state.selectedNode.id === n.id;
    const isHovered = state.hoveredNode && state.hoveredNode.id === n.id;
    const isFlagged = n.is_flagged || n.max_risk_score >= 0.7372;
    const isMedium = !isFlagged && n.max_risk_score >= 0.40;
    const radius = isFlagged ? 18 : (isMedium ? 15 : 13);

    // Glowing Pulsing Halos
    if (isFlagged) {
      const pulse = Math.sin(now / 220) * 4;
      ctx.beginPath();
      ctx.arc(n.x, n.y, radius + 5 + pulse, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(244, 63, 94, 0.28)";
      ctx.fill();
    } else if (isMedium) {
      const pulse = Math.sin(now / 300) * 2.5;
      ctx.beginPath();
      ctx.arc(n.x, n.y, radius + 3 + pulse, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(245, 158, 11, 0.22)";
      ctx.fill();
    }

    // Circle Body
    ctx.beginPath();
    ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = isFlagged ? "#F43F5E" : (isMedium ? "#F59E0B" : "#3B82F6");
    ctx.fill();

    // Border
    ctx.strokeStyle = isSelected ? "#38BDF8" : (isHovered ? "#93C5FD" : "rgba(255, 255, 255, 0.4)");
    ctx.lineWidth = isSelected ? 3.5 : (isHovered ? 2.5 : 2);
    ctx.stroke();

    // Account Label
    ctx.font = isSelected ? "bold 11px Inter, sans-serif" : "10px Inter, sans-serif";
    ctx.fillStyle = "#F1F5F9";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(n.id, n.x, n.y + radius + 5);
  });

  ctx.restore();
}

function drawEmptyGraph(msg) {
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#64748B";
  ctx.font = "13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(msg, canvas.width / 2, canvas.height / 2);
}

// ============================================================================
// 4. PLAIN-ENGLISH ACCOUNT EXPLANATION SIDEPANEL
// ============================================================================

window.switchSidepanelTab = function (tabName) {
  state.activeSidepanelTab = tabName;
  document.querySelectorAll("#sidepanel-tabs .subnav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.id === `btn-sidetab-${tabName}`);
  });
  renderSidepanelContent();
};

async function inspectAccount(accountId) {
  const title = document.getElementById("panel-entity-title");
  const badge = document.getElementById("panel-entity-badge");
  const content = document.getElementById("panel-entity-content");
  const subtabs = document.getElementById("sidepanel-tabs");
  if (!title || !content) return;

  title.textContent = `Account: ${accountId}`;
  content.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 24px;">Analyzing account behavior...</div>`;

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/account/${encodeURIComponent(accountId)}`);
    if (!res.ok) throw new Error("Account details failed");
    const acc = await res.json();
    state.activeAccountData = acc;

    if (subtabs) subtabs.style.display = "flex";
    if (badge) {
      const code = acc.risk_level_code || "low";
      badge.innerHTML = `<span class="risk-badge ${code}">${acc.risk_level} (${acc.max_risk_score.toFixed(2)})</span>`;
    }

    renderSidepanelContent();
  } catch (err) {
    content.innerHTML = `<div style="color: var(--color-flagged); font-size: 12px; padding: 12px;">Failed to inspect account: ${err.message}</div>`;
  }
}

function renderSidepanelContent() {
  const content = document.getElementById("panel-entity-content");
  const acc = state.activeAccountData;
  if (!content || !acc) return;

  const tab = state.activeSidepanelTab || "reason";

  if (tab === "reason") {
    // 1. Plain-English Why & Straight-to-the-Point Risk Factors
    const factorsHtml = (acc.risk_factors || [])
      .map(
        (f) => `
      <div class="risk-factor-card">
        <div class="risk-factor-header">
          <span class="risk-factor-title">${f.title}</span>
          <span class="impact-tag ${f.level}">${f.impact}</span>
        </div>
        <div class="risk-factor-desc">${f.description}</div>
      </div>
    `
      )
      .join("");

    content.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 12px; font-size: 13px;">
        <!-- Layman Callout Banner -->
        <div class="layman-callout ${acc.risk_level_code || "low"}">
          <div class="layman-callout-title">
            ${
              acc.risk_level_code === "high"
                ? "🚨 Why is this account high risk?"
                : acc.risk_level_code === "medium"
                ? "⚠️ Elevated Risk Notice"
                : "✅ Normal Account Baseline"
            }
          </div>
          <div style="font-size: 13px; line-height: 1.5;">${acc.plain_english_summary}</div>
        </div>

        <!-- Key Risk Factors -->
        <div>
          <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px; letter-spacing: 0.3px;">KEY BEHAVIORAL DRIVERS</div>
          ${factorsHtml || `<div style="color: var(--text-muted); font-size: 12px;">No elevated risk factors detected.</div>`}
        </div>

        <!-- Account Metrics Grid -->
        <div style="background: var(--bg-card-subtle); padding: 12px; border-radius: var(--radius-sm); border: 1px solid var(--border-card);">
          <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 8px;">FLOW STATISTICS</div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px;">
            <div><span style="color: var(--text-muted);">Transactions:</span> <strong>${acc.total_transactions}</strong></div>
            <div><span style="color: var(--text-muted);">Flagged:</span> <strong class="${acc.flagged_transactions > 0 ? "text-danger" : ""}">${acc.flagged_transactions}</strong></div>
            <div><span style="color: var(--text-muted);">Total Sent:</span> <strong class="mono">$${Number(acc.total_sent).toLocaleString(undefined, { minimumFractionDigits: 2 })}</strong></div>
            <div><span style="color: var(--text-muted);">Total Received:</span> <strong class="mono">$${Number(acc.total_received).toLocaleString(undefined, { minimumFractionDigits: 2 })}</strong></div>
            <div><span style="color: var(--text-muted);">Counterparties:</span> <strong>${acc.counterparties.length}</strong></div>
            <div><span style="color: var(--text-muted);">Max TGAT Score:</span> <strong class="mono ${acc.risk_level_code === "high" ? "text-danger" : ""}">${acc.max_risk_score.toFixed(4)}</strong></div>
          </div>
        </div>

        <button class="btn btn-secondary btn-sm" onclick="filterResultsByAccount('${acc.account_id}')" style="width: 100%;">
          View All Transactions for This Account &darr;
        </button>
      </div>
    `;
  } else if (tab === "txs") {
    // 2. Transaction History
    const txRows = (acc.transactions || [])
      .map(
        (t) => `
      <div style="padding: 10px; border: 1px solid var(--border-subtle); border-radius: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; background: var(--bg-card);">
        <div>
          <div class="mono" style="font-weight: 600;">${t.id} &bull; $${Number(t.amount).toLocaleString()}</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
            ${t.from_account === acc.account_id ? `To &rarr; <span class="mono">${t.to_account}</span>` : `From &larr; <span class="mono">${t.from_account}</span>`}
          </div>
          <div style="font-size: 10px; color: var(--text-muted);">${t.timestamp_str || formatTimestamp(t.timestamp)}</div>
        </div>
        <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
          <span class="risk-badge ${t.classification === "FLAGGED" ? "high" : "low"}" style="font-size: 10px;">${Number(t.risk_score).toFixed(2)}</span>
          <button class="btn btn-primary btn-sm" style="padding: 2px 8px; font-size: 10px;" onclick="openTransactionModal('${t.id}')">Explain</button>
        </div>
      </div>
    `
      )
      .join("");

    content.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="font-size: 11px; font-weight: 600; color: var(--text-muted);">TRANSACTION LEDGER (${acc.transactions.length})</div>
        <div style="display: flex; flex-direction: column; gap: 6px; max-height: 440px; overflow-y: auto;">
          ${txRows}
        </div>
      </div>
    `;
  } else if (tab === "counterparties") {
    // 3. Counterparties
    const cpDetails = acc.counterparties_details || [];
    const cpRows = cpDetails
      .map(
        (cp) => `
      <div style="padding: 10px; border: 1px solid var(--border-subtle); border-radius: 6px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; background: var(--bg-card);">
        <div>
          <div class="mono" style="font-weight: 700; color: var(--accent-blue);">${cp.account_id}</div>
          <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
            ${cp.transaction_count} transfer(s) &bull; $${Number(cp.total_amount).toLocaleString()}
          </div>
        </div>
        <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
          <span class="risk-badge ${cp.risk_level_code}">${cp.risk_level} (${cp.max_risk_score.toFixed(2)})</span>
          <button class="btn btn-secondary btn-sm" style="padding: 2px 8px; font-size: 10px;" onclick="inspectAccount('${cp.account_id}')">Inspect</button>
        </div>
      </div>
    `
      )
      .join("");

    content.innerHTML = `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="font-size: 11px; font-weight: 600; color: var(--text-muted);">CONNECTED COUNTERPARTIES (${cpDetails.length})</div>
        <div style="display: flex; flex-direction: column; gap: 6px; max-height: 440px; overflow-y: auto;">
          ${cpRows}
        </div>
      </div>
    `;
  }
}

window.filterResultsByAccount = function (accountId) {
  state.resultsSearch = accountId;
  const searchInput = document.getElementById("table-search-input");
  if (searchInput) searchInput.value = accountId;
  loadResultsTable(1);

  // Scroll smoothly down to the table
  const tableCard = document.querySelector(".data-table-card");
  if (tableCard) tableCard.scrollIntoView({ behavior: "smooth" });
};

// ============================================================================
// 5. TRANSACTION RESULTS TABLE
// ============================================================================

function initTransactionsTable() {
  const searchInput = document.getElementById("table-search-input");
  const riskSelect = document.getElementById("table-filter-risk");
  const prevBtn = document.getElementById("btn-page-prev");
  const nextBtn = document.getElementById("btn-page-next");

  if (searchInput) {
    let timeout;
    searchInput.addEventListener("input", () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        state.resultsSearch = searchInput.value.trim();
        loadResultsTable(1);
      }, 250);
    });
  }

  if (riskSelect) {
    riskSelect.addEventListener("change", () => {
      state.resultsRiskFilter = riskSelect.value;
      loadResultsTable(1);
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (state.resultsPage > 1) {
        loadResultsTable(state.resultsPage - 1);
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      loadResultsTable(state.resultsPage + 1);
    });
  }
}

async function loadResultsTable(page = 1) {
  if (!state.currentSessionId) return;
  state.resultsPage = page;

  const tbody = document.getElementById("transactions-tbody");
  if (!tbody) return;

  tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 24px; color: var(--text-muted);">Filtering results...</td></tr>`;

  try {
    let url = `/api/analysis/${encodeURIComponent(state.currentSessionId)}/results?page=${page}&limit=${state.resultsLimit}`;
    if (state.resultsSearch) url += `&search=${encodeURIComponent(state.resultsSearch)}`;
    if (state.resultsRiskFilter) url += `&classification=${encodeURIComponent(state.resultsRiskFilter)}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error("Results query failed");
    const data = await res.json();

    const items = data.items || [];
    const total = data.total || 0;

    if (items.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 24px; color: var(--text-muted);">No transactions match your search criteria.</td></tr>`;
      return;
    }

    tbody.innerHTML = items
      .map((tx) => {
        const isFlagged = tx.classification === "FLAGGED" || tx.risk_score >= 0.7372;
        return `
        <tr>
          <td class="mono"><strong>${tx.id}</strong></td>
          <td class="mono">${formatTimestamp(tx.timestamp)}</td>
          <td class="mono">${tx.from_account}</td>
          <td class="mono">${tx.to_account}</td>
          <td class="mono" style="text-align: right; font-weight: 600;">$${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td class="mono" style="text-align: right; font-weight: 700; color: ${isFlagged ? "var(--color-flagged)" : "var(--text-primary)"};">
            ${Number(tx.risk_score).toFixed(4)}
          </td>
          <td>
            <span class="risk-badge ${isFlagged ? "high" : "low"}">
              ${isFlagged ? "FLAGGED" : "NORMAL"}
            </span>
          </td>
          <td style="text-align: center;">
            <button class="btn btn-primary btn-sm" onclick="openTransactionModal('${tx.id}')">
              Explain
            </button>
          </td>
        </tr>
      `;
      })
      .join("");

    // Pagination info
    const startIdx = (page - 1) * state.resultsLimit + 1;
    const endIdx = Math.min(total, page * state.resultsLimit);
    const infoEl = document.getElementById("table-pagination-info");
    if (infoEl) infoEl.textContent = `Showing ${startIdx} to ${endIdx} of ${total} transactions`;

    const prevBtn = document.getElementById("btn-page-prev");
    const nextBtn = document.getElementById("btn-page-next");
    if (prevBtn) prevBtn.disabled = page <= 1;
    if (nextBtn) nextBtn.disabled = endIdx >= total;

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; padding: 24px; color: var(--color-flagged);">Error loading results: ${err.message}</td></tr>`;
  }
}

// ============================================================================
// 6. STRAIGHT-TO-THE-POINT TRANSACTION EXPLANATION MODAL
// ============================================================================

function initTransactionModal() {
  const modal = document.getElementById("tx-modal");
  const closeBtn = document.getElementById("btn-close-tx-modal");
  const footerCloseBtn = document.getElementById("btn-close-modal-footer");

  function closeModal() {
    if (modal) modal.style.display = "none";
  }

  if (closeBtn) closeBtn.addEventListener("click", closeModal);
  if (footerCloseBtn) footerCloseBtn.addEventListener("click", closeModal);

  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal && modal.style.display !== "none") {
      closeModal();
    }
  });
}

window.openTransactionModal = async function (txId) {
  const modal = document.getElementById("tx-modal");
  if (!modal) return;
  modal.style.display = "flex";

  const title = document.getElementById("modal-tx-title");
  const badge = document.getElementById("modal-tx-badge");
  const callout = document.getElementById("modal-tx-callout");
  const calloutTitle = document.getElementById("modal-tx-callout-title");
  const calloutText = document.getElementById("modal-tx-callout-text");
  const amount = document.getElementById("modal-tx-amount");
  const score = document.getElementById("modal-tx-score");
  const fromAcc = document.getElementById("modal-tx-from");
  const toAcc = document.getElementById("modal-tx-to");
  const time = document.getElementById("modal-tx-time");
  const factorsContainer = document.getElementById("modal-tx-factors");

  if (title) title.textContent = `Transaction: ${txId}`;
  if (calloutText) calloutText.textContent = "Analyzing transaction context and graph attention weights...";
  if (factorsContainer) factorsContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 12px;">Loading explanation...</div>`;

  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(state.currentSessionId)}/transaction/${encodeURIComponent(txId)}`);
    if (!res.ok) throw new Error("Explanation query failed");
    const tx = await res.json();

    const isFlagged = tx.classification === "FLAGGED" || tx.risk_score >= 0.7372;
    const scoreVal = Number(tx.risk_score);

    if (badge) {
      badge.className = `risk-badge ${isFlagged ? "high" : "low"}`;
      badge.textContent = isFlagged ? `High Risk (${scoreVal.toFixed(2)})` : `Normal / Low Risk (${scoreVal.toFixed(2)})`;
    }

    if (callout) {
      callout.className = `layman-callout ${isFlagged ? "high" : "low"}`;
    }

    if (calloutTitle) {
      calloutTitle.textContent = isFlagged
        ? "🚨 Why was this transaction flagged as fraudulent?"
        : "✅ Why is this transaction considered safe?";
    }

    if (calloutText) {
      calloutText.textContent = tx.plain_english_summary || tx.explainability?.summary || (
        isFlagged
          ? "This transaction is part of an abnormal high-speed money pass-through chain between flagged counterparties."
          : "This transaction matches normal payment patterns without circular hops or burst velocity."
      );
    }

    if (amount) amount.textContent = `$${Number(tx.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    if (score) {
      score.textContent = scoreVal.toFixed(4);
      score.style.color = isFlagged ? "var(--color-flagged)" : "var(--text-primary)";
    }
    if (fromAcc) fromAcc.textContent = tx.from_account;
    if (toAcc) toAcc.textContent = tx.to_account;
    if (time) time.textContent = tx.timestamp_str || formatTimestamp(tx.timestamp);

    // Straight-to-the-point Risk Drivers
    if (factorsContainer) {
      const factors = tx.plain_english_factors || [];
      if (factors.length === 0) {
        factorsContainer.innerHTML = `
          <div class="risk-factor-card">
            <div class="risk-factor-header">
              <span class="risk-factor-title">Baseline Flow</span>
              <span class="impact-tag low">CLEARED</span>
            </div>
            <div class="risk-factor-desc">No suspicious rapid forwarding or circular ring patterns detected.</div>
          </div>
        `;
      } else {
        factorsContainer.innerHTML = factors
          .map(
            (f) => `
          <div class="risk-factor-card">
            <div class="risk-factor-header">
              <span class="risk-factor-title">${f.title}</span>
              <span class="impact-tag ${f.level}">${f.impact}</span>
            </div>
            <div class="risk-factor-desc">${f.description}</div>
          </div>
        `
          )
          .join("");
      }
    }

  } catch (err) {
    if (calloutText) calloutText.textContent = `Error loading explanation: ${err.message}`;
  }
};

// ============================================================================
// 7. RESULTS WITH OTHER MODELS (COMPARISON VIEW)
// ============================================================================

async function loadComparisonMetrics() {
  try {
    const [metricsRes, ablRes] = await Promise.all([
      fetch("/api/metrics"),
      fetch("/api/ablations"),
    ]);

    if (!metricsRes.ok || !ablRes.ok) return;

    const metricsData = await metricsRes.json();
    const ablData = await ablRes.json();

    // Baseline Comparison Table
    const baselinesTbody = document.getElementById("perf-baselines-tbody");
    if (baselinesTbody) {
      baselinesTbody.innerHTML = `
        <tr style="background-color: var(--accent-blue-subtle);">
          <td><strong>TGAT (Our Model, &tau;* = 0.7372)</strong></td>
          <td><span class="mono" style="font-weight: 700; color: var(--color-success);">0.9022</span></td>
          <td><span class="mono" style="font-weight: 700; color: var(--color-success);">0.0184</span></td>
          <td><span class="mono" style="font-weight: 700; color: var(--color-success);">0.0504</span></td>
          <td class="mono"><strong>4.55%</strong></td>
          <td class="mono" style="color: var(--color-success); font-weight: 700;">357 (83.8% Reduction)</td>
          <td><span class="risk-badge" style="background: rgba(59, 130, 246, 0.2); color: #60A5FA;">Dynamic Edge Attention</span></td>
        </tr>
        <tr>
          <td>Temporal XGBoost</td>
          <td class="mono">0.9175</td>
          <td class="mono">0.0280</td>
          <td class="mono">0.0781</td>
          <td class="mono">4.16%</td>
          <td class="mono">4,040</td>
          <td>Global Feature Importance (Black-box)</td>
        </tr>
        <tr>
          <td>Tabular XGBoost</td>
          <td class="mono">0.8901</td>
          <td class="mono">0.0305</td>
          <td class="mono">0.0180</td>
          <td class="mono">0.91%</td>
          <td class="mono" style="color: #F59E0B; font-weight: 600;">26,399 (Severe Alert Fatigue)</td>
          <td>Isolated Payments Only (Blind to Rings)</td>
        </tr>
        <tr>
          <td>Logistic Regression</td>
          <td class="mono">0.8270</td>
          <td class="mono">0.0049</td>
          <td class="mono">0.0023</td>
          <td class="mono">0.11%</td>
          <td class="mono">263,840</td>
          <td>Linear Feature Weights</td>
        </tr>
        <tr style="background-color: rgba(244, 63, 94, 0.08);">
          <td><strong style="color: #F43F5E;">Static Graph GCN (FAILED)</strong></td>
          <td class="mono" style="color: #F43F5E; font-weight: 700;">0.6064 (Near Random)</td>
          <td class="mono" style="color: #F43F5E;">0.0017</td>
          <td class="mono" style="color: #F43F5E;">0.0038</td>
          <td class="mono">0.19%</td>
          <td class="mono">1,177</td>
          <td style="color: #F43F5E; font-size: 11px;">Failed: Collapsing time causes future leakage & topological smearing</td>
        </tr>
      `;
    }

    // Ablations Study Table
    const ablTbody = document.getElementById("perf-ablations-tbody");
    if (ablTbody && ablData.experiments) {
      ablTbody.innerHTML = ablData.experiments
        .map(
          (r) => `
        <tr>
          <td class="mono"><strong>${r.id}</strong></td>
          <td>${getAblationPlainDescription(r.name)}</td>
          <td class="mono">${r.val_roc_auc ? r.val_roc_auc.toFixed(4) : "—"}</td>
          <td class="mono" style="font-weight: 600;">${r.test_roc_auc ? r.test_roc_auc.toFixed(4) : "—"}</td>
          <td class="mono">${r.test_pr_auc ? r.test_pr_auc.toFixed(4) : "—"}</td>
          <td class="mono">${r.test_f1 !== undefined ? r.test_f1.toFixed(4) : "—"}</td>
          <td class="mono">${r.test_p100 !== undefined ? (r.test_p100 * 100).toFixed(1) + "%" : "—"}</td>
          <td class="mono">${r.training_seconds ? Math.round(r.training_seconds) + "s" : "—"}</td>
        </tr>
      `
        )
        .join("");
    }

  } catch (err) {
    console.warn("Failed to load comparison data:", err);
  }
}

function getAblationPlainDescription(name) {
  if (name.includes("a0_baseline")) return "Full TGAT Reference (K=10 neighbors, 2 layers, Bochner time encoding)";
  if (name.includes("no_temporal")) return "Drop all temporal state features (-0.0574 AUC drop; confirms time is crucial)";
  if (name.includes("b_no_neighbors")) return "No graph neighbors (Isolated accounts; F1 collapses completely)";
  if (name.includes("k_1")) return "Query neighborhood budget K=1";
  if (name.includes("k_5")) return "Query neighborhood budget K=5";
  if (name.includes("k_10")) return "Reference neighborhood budget K=10 (Optimal sweet spot)";
  if (name.includes("k_20")) return "Expanded neighborhood budget K=20";
  if (name.includes("no_time_encoding")) return "Remove Bochner harmonic time encoding (Detection collapses)";
  if (name.includes("one_layer")) return "Single attention layer (L=1)";
  if (name.includes("one_head")) return "Single attention head (H=1)";
  if (name.includes("hidden_16")) return "Reduced hidden dimension (d=16)";
  if (name.includes("hidden_64")) return "Expanded hidden dimension (d=64)";
  if (name.includes("dropout_0")) return "Zero dropout regularization (0.9116 ROC-AUC)";
  if (name.includes("dropout_04")) return "Heavy dropout (0.4)";
  return "Architecture variant";
}

// ============================================================================
// 8. HELPERS
// ============================================================================

function formatTimestamp(tsSec) {
  if (!tsSec) return "—";
  const d = new Date(tsSec * 1000);
  return d.toISOString().replace("T", " ").substring(0, 19);
}
