const SPRITE = "https://play.pokemonshowdown.com/sprites";
const BALL = `${SPRITE}/itemicons/pokeball.png`;

const state = {
  offset: 0,
  limit: 80,
  total: 0,
  filters: { q: "", result: "", format: "" },
  formats: [],
  replays: [],
  activeId: null,
  detail: null,
  turn: 0,
  scores: null,
  loadedModels: [],
};

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = j.detail || JSON.stringify(j);
    } catch (_) {
      try {
        msg = await res.text();
      } catch (__) {}
    }
    throw new Error(msg);
  }
  return res.json();
}

function replaySpriteGen() {
  const fmt = (state.detail?.meta?.format || "").toLowerCase();
  const m = fmt.match(/gen(\d+)/);
  const gen = m ? Number(m[1]) : 5;
  return Number.isFinite(gen) && gen > 0 ? gen : 5;
}

function spriteUrl(base, back = false) {
  const id = (base || "unknown").toLowerCase().replace(/[^a-z0-9]/g, "");
  const gen = replaySpriteGen();
  // Gens 7+ → animated; 1–5 → matching gen folders; gen6 has fronts only
  // (backs fall back to gen5-back, matching common PS usage).
  if (gen >= 7) {
    const folder = back ? "ani-back" : "ani";
    return `${SPRITE}/${folder}/${id}.gif`;
  }
  if (gen === 6) {
    return back
      ? `${SPRITE}/gen5-back/${id}.png`
      : `${SPRITE}/gen6/${id}.png`;
  }
  const g = Math.min(Math.max(gen, 1), 5);
  // Gen1 sheets have large empty canvas padding; PS offsets with spriteData.y
  // in its 3D scene. Serve trimmed proxies so art sits on our pedestal.
  if (g === 1) {
    const folder = back ? "gen1-back" : "gen1";
    return `/api/sprites/${folder}/${id}.png`;
  }
  const folder = back ? `gen${g}-back` : `gen${g}`;
  return `${SPRITE}/${folder}/${id}.png`;
}

function piconUrl(base) {
  const id = (base || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  if (!id) return BALL;
  const gen = replaySpriteGen();
  const g = gen >= 7 ? 5 : Math.min(gen, 6);
  return `${SPRITE}/gen${g}/${id}.png`;
}

function hpPercent(hpFrac) {
  // UniversalPokemon.hp_pct is a fraction in [0, 1].
  const pct = Number(hpFrac) * 100;
  if (!Number.isFinite(pct)) return 0;
  return Math.max(0, Math.min(100, pct));
}

function setHp(el, hpFrac) {
  const bar = el.querySelector(".bar > span");
  const clamped = hpPercent(hpFrac);
  bar.style.width = `${clamped}%`;
  el.classList.remove("low", "critical");
  if (clamped <= 20) el.classList.add("critical");
  else if (clamped <= 50) el.classList.add("low");
}

function renderParty(el, party, activeBase) {
  el.innerHTML = "";
  const slots = [...(party || [])];
  while (slots.length < 6) slots.push(null);
  const active = (activeBase || "").toLowerCase();
  slots.slice(0, 6).forEach((p) => {
    const slot = document.createElement("div");
    const isActive =
      p &&
      (p.active ||
        (p.base_species || p.name || "").toLowerCase() === active);
    const isFainted = !!(p && p.fainted);
    slot.className =
      "slot" +
      (p ? "" : " empty") +
      (isFainted ? " fainted" : "") +
      (isActive && !isFainted ? " active" : "");
    if (p) {
      slot.title = isFainted ? `${p.name} (KO)` : p.name;
    }
    if (p) {
      const img = document.createElement("img");
      img.src = piconUrl(p.base_species || p.name);
      img.alt = p.name;
      img.onerror = () => {
        img.src = BALL;
      };
      slot.appendChild(img);
      if (isFainted) {
        const ko = document.createElement("span");
        ko.className = "ko-mark";
        ko.textContent = "KO";
        slot.appendChild(ko);
      }
    }
    el.appendChild(slot);
  });
}

function setStatusChip(el, status) {
  const s = (status || "").trim();
  if (!s || s === "none" || s === "nostatus") {
    el.hidden = true;
    el.textContent = "";
    el.className = "status-chip";
    return;
  }
  const key = s.toLowerCase().replace(/[^a-z]/g, "").slice(0, 3);
  el.hidden = false;
  el.textContent = s.toUpperCase().slice(0, 3);
  el.className = `status-chip ${key}`;
}

function setWillUse(el, action) {
  const label = (action || "").trim();
  if (!label) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = `<span class="will-prefix">Will Use</span><span class="will-move">${label}</span>`;
}

function normalizeCondition(value) {
  // UniversalState stores weather / side conditions as a single string
  // (e.g. "noconditions", "stealthrock"). Guard against accidental list/char splits.
  if (value == null) return "";
  if (Array.isArray(value)) return value.join("");
  return String(value).trim();
}

function isBlankCondition(value) {
  const t = normalizeCondition(value).toLowerCase().replace(/,/g, "");
  return (
    !t ||
    t === "none" ||
    t === "noweather" ||
    t === "nocond" ||
    t === "noconditions" ||
    t === "nofield"
  );
}

function renderConditionChips(el, turn) {
  el.innerHTML = "";
  const tags = [];
  const weather = normalizeCondition(turn.weather);
  const yours = normalizeCondition(turn.player_conditions);
  const foe = normalizeCondition(turn.opponent_conditions);
  if (!isBlankCondition(weather)) {
    tags.push({ text: weather, warn: false });
  }
  if (turn.forced_switch) tags.push({ text: "forced switch", warn: true });
  if (!isBlankCondition(yours)) {
    tags.push({ text: yours, warn: false });
  }
  if (!isBlankCondition(foe)) {
    tags.push({ text: `foe ${foe}`, warn: false });
  }
  if (!tags.length) {
    const span = document.createElement("span");
    span.className = "chip-tag";
    span.textContent = "clear field";
    el.appendChild(span);
    return;
  }
  tags.slice(0, 6).forEach((t) => {
    const span = document.createElement("span");
    span.className = "chip-tag" + (t.warn ? " warn" : "");
    span.textContent = t.text;
    el.appendChild(span);
  });
}

function renderModelTray() {
  const tray = document.getElementById("modelTray");
  tray.innerHTML = "";
  if (!state.loadedModels.length) {
    const span = document.createElement("span");
    span.className = "status-line";
    span.textContent = "No models loaded";
    tray.appendChild(span);
    return;
  }
  state.loadedModels.forEach((m) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const ck =
      m.checkpoint === -1 || m.checkpoint === null
        ? ""
        : `@${m.checkpoint}`;
    const temp =
      m.temperature != null && Math.abs(Number(m.temperature) - 1) > 1e-6
        ? ` t=${Number(m.temperature).toFixed(2)}`
        : "";
    chip.innerHTML = `<span>${m.name}${ck}${temp}</span>`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.title = "Unload";
    btn.textContent = "×";
    btn.onclick = () => unloadModel(m.key);
    chip.appendChild(btn);
    tray.appendChild(chip);
  });
}

function renderReplayList() {
  const list = document.getElementById("replayList");
  list.innerHTML = "";
  state.replays.forEach((r) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "replay-row" + (r.id === state.activeId ? " active" : "");
    const badge =
      r.result === "WIN" || r.result === "LOSS"
        ? `<span class="badge ${r.result.toLowerCase()}">${r.result}</span>`
        : "";
    btn.innerHTML = `
      <div class="vs">${r.player} vs ${r.opponent}</div>
      <div class="meta">${badge}<span>${r.format}</span><span>${r.date}</span>${
      r.rating != null ? `<span>${r.rating}</span>` : ""
    }</div>`;
    btn.onclick = () => openReplay(r.id);
    list.appendChild(btn);
  });
  const start = state.total ? state.offset + 1 : 0;
  const end = Math.min(state.offset + state.limit, state.total);
  document.getElementById("pageInfo").textContent = `${start}–${end} / ${state.total}`;
  document.getElementById("replayCount").textContent = `${state.total} total`;
  document.getElementById("prevPage").disabled = state.offset <= 0;
  document.getElementById("nextPage").disabled =
    state.offset + state.limit >= state.total;
}

function renderTurn() {
  if (!state.detail) return;
  const t = state.detail.turns[state.turn];
  if (!t) return;

  document.getElementById("turnBadge").textContent = `TURN ${t.turn + 1}`;
  document.getElementById("playerName").textContent = state.detail.meta.player;
  document.getElementById("oppName").textContent = state.detail.meta.opponent;
  document.getElementById("playerSpecies").textContent = t.player_active;
  document.getElementById("oppSpecies").textContent = t.opponent_active;
  document.getElementById("playerHpPct").textContent = `${Math.round(
    hpPercent(t.player_hp_pct)
  )}%`;
  document.getElementById("oppHpPct").textContent = `${Math.round(
    hpPercent(t.opponent_hp_pct)
  )}%`;
  setHp(document.getElementById("playerHp"), t.player_hp_pct);
  setHp(document.getElementById("oppHp"), t.opponent_hp_pct);
  setStatusChip(document.getElementById("playerStatus"), t.player_status);
  setStatusChip(document.getElementById("oppStatus"), t.opponent_status);

  const ps = document.getElementById("playerSprite");
  const os = document.getElementById("oppSprite");
  ps.src = spriteUrl(t.player_active_base, true);
  os.src = spriteUrl(t.opponent_active_base, false);
  ps.onerror = () => {
    ps.src = BALL;
  };
  os.onerror = () => {
    os.src = BALL;
  };

  setWillUse(document.getElementById("playerWillUse"), t.player_will_use);
  setWillUse(document.getElementById("oppWillUse"), t.opponent_will_use);

  renderConditionChips(document.getElementById("conditions"), t);
  // Prefer per-turn snapshots; never fall back to end-of-battle roster for KOs.
  renderParty(
    document.getElementById("playerParty"),
    Array.isArray(t.player_party) ? t.player_party : [],
    t.player_active_base
  );
  renderParty(
    document.getElementById("oppParty"),
    Array.isArray(t.opponent_party) ? t.opponent_party : [],
    t.opponent_active_base
  );

  const slider = document.getElementById("turnSlider");
  slider.max = String(Math.max(0, state.detail.num_turns - 1));
  slider.value = String(state.turn);

  renderAnalysis();
}

function renderAnalysis() {
  const root = document.getElementById("analysis");
  root.innerHTML = "";
  if (!state.detail) {
    root.innerHTML = `<div class="empty-state">Open a replay to see distributions.</div>`;
    return;
  }
  const t = state.detail.turns[state.turn];
  const gt = document.createElement("div");
  gt.className = "gt-banner";
  gt.innerHTML = t.missing
    ? `<strong>Chosen:</strong> unrevealed`
    : `<strong>Chosen:</strong> ${t.chosen_label}`;
  root.appendChild(gt);

  if (!state.scores || !state.scores.models || !state.scores.models.length) {
    const tip = document.createElement("div");
    tip.className = "empty-state";
    tip.textContent = state.loadedModels.length
      ? "Scoring…"
      : "Load a model to compare action probabilities.";
    root.appendChild(tip);
    return;
  }

  if (state.scores.long_battle && state.turn === 0) {
    const w = document.createElement("div");
    w.className = "warn";
    w.textContent =
      state.scores.message ||
      "Long battle: scored with sliding KV-cache context (ladder-matched).";
    root.appendChild(w);
  }
  state.scores.models.forEach((m) => {
    const turn = (m.turns || [])[state.turn];
    const card = document.createElement("div");
    card.className = "model-card";
    const agreeClass =
      !turn || turn.missing || turn.agree == null
        ? "na"
        : turn.agree
        ? "yes"
        : "no";
    const agreeText =
      !turn || turn.missing || turn.agree == null
        ? "n/a"
        : turn.agree
        ? "agree"
        : "disagree";
    card.innerHTML = `<header>
      <span class="name">${m.model_key}</span>
      <span class="agree ${agreeClass}">${agreeText}</span>
    </header>`;
    const bars = document.createElement("div");
    bars.className = "bars";
    if (turn && turn.legal) {
      turn.legal.forEach((idx, i) => {
        const row = document.createElement("div");
        row.className = "bar-row" + (turn.gt === idx ? " gt" : "");
        const pct = (turn.probs[i] || 0) * 100;
        const label = turn.labels[i] || `action ${idx}`;
        const lab = document.createElement("div");
        lab.className = "lab";
        lab.textContent = label;
        lab.title = label;
        const track = document.createElement("div");
        track.className = "track";
        const fill = document.createElement("span");
        fill.style.width = `${pct}%`;
        track.appendChild(fill);
        const pctEl = document.createElement("div");
        pctEl.className = "pct";
        pctEl.textContent = `${pct.toFixed(1)}%`;
        row.appendChild(lab);
        row.appendChild(track);
        row.appendChild(pctEl);
        bars.appendChild(row);
      });
    }
    card.appendChild(bars);
    root.appendChild(card);
  });
}

async function refreshHealth() {
  const h = await api("/api/health");
  state.formats = h.formats || [];
  const sel = document.getElementById("filterFormat");
  const cur = sel.value;
  sel.innerHTML = `<option value="">Format</option>`;
  state.formats.forEach((f) => {
    const o = document.createElement("option");
    o.value = f;
    o.textContent = f;
    sel.appendChild(o);
  });
  if (cur) sel.value = cur;
}

async function refreshModels() {
  const avail = await api("/api/models/available");
  const sel = document.getElementById("modelSelect");
  sel.innerHTML = "";
  (avail.names || []).forEach((n) => {
    const o = document.createElement("option");
    o.value = n;
    o.textContent = n;
    sel.appendChild(o);
  });
  const loaded = await api("/api/models/loaded");
  state.loadedModels = loaded.models || [];
  renderModelTray();
}

async function loadReplays() {
  const params = new URLSearchParams({
    offset: String(state.offset),
    limit: String(state.limit),
  });
  if (state.filters.q) params.set("q", state.filters.q);
  if (state.filters.result) params.set("result", state.filters.result);
  if (state.filters.format) params.set("format", state.filters.format);
  const data = await api(`/api/replays?${params}`);
  state.total = data.total;
  state.replays = data.items || [];
  renderReplayList();
}

async function openReplay(id) {
  state.activeId = id;
  state.scores = null;
  document.getElementById("scoreStatus").textContent = "Loading replay…";
  renderReplayList();
  const detail = await api(`/api/replays/${encodeURIComponent(id)}`);
  state.detail = detail;
  state.turn = 0;
  document.getElementById("emptyBattle").hidden = true;
  document.getElementById("battleView").hidden = false;
  renderTurn();
  await rescore();
}

function needsLongBattleScoring() {
  if (!state.detail || !state.loadedModels.length) return false;
  // Obs length = decision turns + terminal frame.
  const tObs = (state.detail.num_turns || 0) + 1;
  return state.loadedModels.some((m) => tObs > (m.max_seq_len || 128));
}

async function rescore() {
  if (!state.activeId) return;
  if (!state.loadedModels.length) {
    state.scores = { models: [] };
    document.getElementById("scoreStatus").textContent = "";
    renderAnalysis();
    return;
  }
  const longBattle = needsLongBattleScoring();
  document.getElementById("scoreStatus").textContent = longBattle
    ? "This is a long battle, please wait…"
    : "Scoring models…";
  try {
    state.scores = await api(
      `/api/score/${encodeURIComponent(state.activeId)}`
    );
    const n = (state.scores.models || []).length;
    const rolled = (state.scores.models || []).filter((m) => m.rolled_context)
      .length;
    document.getElementById("scoreStatus").textContent =
      rolled > 0
        ? `Scored ${n} model(s) (${rolled} with sliding KV context)`
        : `Scored ${n} model(s)`;
  } catch (e) {
    document.getElementById("scoreStatus").textContent = `Score error: ${e.message}`;
    state.scores = { models: [] };
  }
  renderAnalysis();
}

function syncTempLabel() {
  const el = document.getElementById("tempInput");
  const label = document.getElementById("tempValue");
  if (!el || !label) return;
  label.textContent = Number(el.value).toFixed(2);
}

async function loadModel() {
  const name = document.getElementById("modelSelect").value;
  const ckRaw = document.getElementById("ckptInput").value.trim();
  const temperature = Number(document.getElementById("tempInput").value);
  const body = { name, temperature };
  if (ckRaw !== "") body.checkpoint = Number(ckRaw);
  document.getElementById("loadModelBtn").disabled = true;
  try {
    await api("/api/models/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await refreshModels();
    await rescore();
  } catch (e) {
    alert(`Load failed: ${e.message}`);
  } finally {
    document.getElementById("loadModelBtn").disabled = false;
  }
}

async function unloadModel(key) {
  try {
    await api(`/api/models/${encodeURIComponent(key)}`, { method: "DELETE" });
    await refreshModels();
    await rescore();
  } catch (e) {
    alert(`Unload failed: ${e.message}`);
  }
}

function bind() {
  document.getElementById("loadModelBtn").onclick = loadModel;
  const tempInput = document.getElementById("tempInput");
  if (tempInput) {
    tempInput.oninput = syncTempLabel;
    syncTempLabel();
  }
  document.getElementById("prevPage").onclick = () => {
    state.offset = Math.max(0, state.offset - state.limit);
    loadReplays();
  };
  document.getElementById("nextPage").onclick = () => {
    if (state.offset + state.limit < state.total) {
      state.offset += state.limit;
      loadReplays();
    }
  };
  let searchTimer;
  document.getElementById("searchQ").oninput = (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.filters.q = e.target.value.trim();
      state.offset = 0;
      loadReplays();
    }, 250);
  };
  document.getElementById("filterResult").onchange = (e) => {
    state.filters.result = e.target.value;
    state.offset = 0;
    loadReplays();
  };
  document.getElementById("filterFormat").onchange = (e) => {
    state.filters.format = e.target.value;
    state.offset = 0;
    loadReplays();
  };
  document.getElementById("prevTurn").onclick = () => {
    if (!state.detail) return;
    state.turn = Math.max(0, state.turn - 1);
    renderTurn();
  };
  document.getElementById("nextTurn").onclick = () => {
    if (!state.detail) return;
    state.turn = Math.min(state.detail.num_turns - 1, state.turn + 1);
    renderTurn();
  };
  document.getElementById("turnSlider").oninput = (e) => {
    state.turn = Number(e.target.value);
    renderTurn();
  };
  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "ArrowLeft") {
      document.getElementById("prevTurn").click();
    } else if (e.key === "ArrowRight") {
      document.getElementById("nextTurn").click();
    }
  });
}

async function init() {
  bind();
  renderAnalysis();
  await refreshHealth();
  await refreshModels();
  await loadReplays();
}

init().catch((e) => {
  console.error(e);
  alert(`Failed to start UI: ${e.message}`);
});
