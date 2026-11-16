import { Action, ACTION_LABELS, LANCE_LABELS, MusGame, cardImage } from './mus-engine.js';
import { LstmPolicy } from './policy.js';

// Modelos exportados con scripts/export_web_policy.py.
// `stats`: % de victorias en 300 partidas (scripts/evaluate.py).
const MODELS = [
  { id: 'ppo_vs_heuristic', label: 'PPO vs heurístico · 1M pasos', stats: 'Gana 64% al bot heurístico y 60% al Sheriff' },
  { id: 'ppo_colab_selfplay', label: 'Currículum + self-play · 7M pasos', stats: 'Gana 58% al bot heurístico y 46% al Sheriff' },
  { id: 'ppo_LinearDecay', label: 'Ajuste fino (LR lineal) · 8.5M pasos', stats: 'Gana 55% al bot heurístico y 39% al Sheriff' },
];
const HUMAN = 0, AI = 1;
const AI_DELAY_MS = 850;
const VALUE_NAMES = { 10: 'reyes', 9: 'caballos', 8: 'sotas', 7: 'sietes', 6: 'seises', 5: 'cincos', 4: 'cuatros', 1: 'ases' };
const KEYS = { p: Action.PASO, e: Action.ENVIDO, o: Action.ORDAGO, q: Action.QUIERO, n: Action.NO_QUIERO };

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const c of [].concat(children)) node.append(c);
  return node;
};

const state = {
  game: null,
  policy: null,
  policyCache: new Map(),
  aiEpisodeStart: true,
  reveal: null,          // { title, hands, points, gameOver }
  handPoints: [],
  aiTimer: null,
  showAiCards: false,
  dealt: false,
};

// ------------------------------------------------------------ carga del modelo
async function loadPolicy(id) {
  if (!state.policyCache.has(id)) {
    const policy = await LstmPolicy.load(`models/${id}`);
    const test = policy.selfTest();
    console.info(`[policy] ${id}: test PyTorch↔JS ${test.ok ? 'OK' : 'FALLO'} (error máx ${test.maxErr.toExponential(2)})`);
    state.policyCache.set(id, policy);
  }
  return state.policyCache.get(id);
}

async function newGame() {
  clearTimeout(state.aiTimer);
  const id = $('model-select').value;
  setStatus('Cargando la IA…');
  try {
    state.policy = await loadPolicy(id);
  } catch (err) {
    setStatus('No se pudo cargar el modelo');
    console.error(err);
    return;
  }
  state.policy.reset();
  state.aiEpisodeStart = true;
  state.reveal = null;
  state.handPoints = [];
  $('log').replaceChildren();
  $('probs').replaceChildren(el('p', { className: 'muted', textContent: 'Aún no ha jugado.' }));
  $('think-note').textContent = '';
  $('think-lance').textContent = '';
  state.game = new MusGame(30);
  log(`Nueva partida contra ${MODELS.find((m) => m.id === id).label}`, 'sys');
  processEvents(state.game.reset());
  state.dealt = true;
  render();
  scheduleAi();
}

// ------------------------------------------------------------ turnos
function humanPlay(action) {
  const g = state.game;
  if (!g || state.reveal || g.gameOver || g.current !== HUMAN) return;
  if (!g.legalActions().includes(action)) return;
  processEvents(g.step(action));
  render();
  scheduleAi();
}

function scheduleAi() {
  clearTimeout(state.aiTimer);
  const g = state.game;
  if (!g || state.reveal || g.gameOver || g.current !== AI) return;
  state.aiTimer = setTimeout(aiPlay, AI_DELAY_MS);
}

function aiPlay() {
  const g = state.game;
  if (!g || state.reveal || g.gameOver || g.current !== AI) return;
  const legal = g.legalActions();
  const lance = g.lance;
  const { action, probs } = state.policy.act(g.observation(AI), {
    episodeStart: state.aiEpisodeStart, deterministic: true,
  });
  state.aiEpisodeStart = false;
  const events = g.step(action);
  const played = events.find((e) => e.kind === 'action').action;
  showThinking(probs, legal, action, played, lance);
  showBubble(ACTION_LABELS[played]);
  processEvents(events);
  render();
  scheduleAi();
}

// ------------------------------------------------------------ eventos del motor
function processEvents(events) {
  const g = state.game;
  for (const ev of events) {
    switch (ev.kind) {
      case 'new_hand':
        state.handPoints = [];
        log(`Mano ${ev.number} · ${ev.mano === HUMAN ? 'eres mano' : 'la IA es mano'}`, 'sys');
        state.dealt = true;
        break;
      case 'action': {
        const who = ev.player === HUMAN ? 'Tú' : 'IA';
        log(`${who}: ${ACTION_LABELS[ev.action]}`, ev.player === HUMAN ? 'h' : 'a');
        break;
      }
      case 'skip':
        if (ev.who === null) log(`${LANCE_LABELS[ev.lance]}: nadie tiene, no se juega`, '');
        else log(`${LANCE_LABELS[ev.lance]}: solo ${ev.who === HUMAN ? 'tú tienes' : 'la IA tiene'}`, '');
        break;
      case 'points':
        state.handPoints.push(ev);
        if (ev.reason === 'no_quiero' || ev.reason === 'solo') {
          log(`  +${ev.points} para ${ev.player === HUMAN ? 'ti' : 'la IA'} (${pointsReason(ev)})`, ev.player === HUMAN ? 'h' : 'a');
        }
        break;
      case 'ordago':
        log(`¡Órdago aceptado a ${LANCE_LABELS[ev.lance].toLowerCase()}!`, 'sys');
        state.handPoints.push({ kind: 'points', player: ev.winner, points: null, lance: ev.lance, reason: 'ordago' });
        break;
      case 'hand_end':
        state.reveal = {
          reason: ev.reason, hands: ev.hands, mano: ev.mano,
          points: state.handPoints, scores: ev.scores, gameOver: g.gameOver, winner: g.winner,
        };
        break;
      case 'game_over':
        log(ev.winner === HUMAN ? '¡Has ganado la partida!' : 'La IA gana la partida', 'sys');
        break;
      default:
    }
  }
}

function pointsReason(ev) {
  const lance = LANCE_LABELS[ev.lance];
  const owner = state.reveal?.hands?.[ev.player] ?? null;
  switch (ev.reason) {
    case 'no_quiero': return `${lance}: no quiso`;
    case 'solo': return `${lance}: solo uno tiene`;
    case 'paso': return `${lance} en paso`;
    case 'envido': return `${lance}: envite visto`;
    case 'ordago': return `Órdago a ${lance.toLowerCase()}`;
    case 'bonus':
      if (ev.lance === 2) return owner ? describePares(owner) : 'Pares';
      return owner ? (owner.tieneJuego ? `Juego de ${owner.juego}` : `Punto (${owner.juego})`) : 'Juego';
    default: return lance;
  }
}

// ------------------------------------------------------------ render
function render() {
  const g = state.game;
  if (!g) return;
  const r = state.reveal;
  const hands = r ? r.hands : g.hands;
  const mano = r ? r.mano : g.mano;

  renderCards($('human-cards'), hands[HUMAN].cards, false);
  renderCards($('ai-cards'), hands[AI].cards, !(state.showAiCards || r));
  state.dealt = false;
  $('human-mano').hidden = mano !== HUMAN;
  $('ai-mano').hidden = mano !== AI;
  renderHandInfo(hands[HUMAN]);

  // Lances
  $('lances').replaceChildren(...LANCE_LABELS.map((name, i) =>
    el('li', { className: r ? 'done' : g.lanceStatus[i] === 'pending' ? '' : g.lanceStatus[i], textContent: name })));

  // Estado y apuesta
  const statusEl = $('status');
  statusEl.classList.remove('thinking');
  if (g.gameOver) setStatus(g.winner === HUMAN ? '¡Victoria!' : 'Derrota');
  else if (r) setStatus('Fin de la mano');
  else if (g.current === HUMAN) setStatus('Tu turno');
  else { setStatus('La IA está pensando'); statusEl.classList.add('thinking'); }

  const bet = $('bet');
  bet.classList.remove('ordago');
  if (!r && !g.gameOver && g.lanceState === 1) {
    const who = g.better === HUMAN ? 'Has' : 'La IA ha';
    if (g.bet >= g.target) { bet.textContent = `¡${who} lanzado un órdago!`; bet.classList.add('ordago'); }
    else bet.textContent = `${who} envidado ${g.bet}`;
  } else bet.textContent = '';

  renderActions();
  renderScores(g.scores);
  renderOverlay();
}

function renderCards(container, cards, faceDown) {
  const nodes = cards.map((c, i) => {
    const d = el('div', { className: `card${faceDown ? ' back' : ''}${state.dealt ? ' deal' : ''}` });
    d.style.backgroundImage = `url(${faceDown ? 'cards/dorso.webp' : cardImage(c)})`;
    d.style.animationDelay = `${i * 60}ms`;
    d.setAttribute('role', 'img');
    d.setAttribute('aria-label', faceDown ? 'Carta boca abajo' : `${c.rango} de ${c.palo}`);
    return d;
  });
  container.replaceChildren(...nodes);
}

function describePares(h) {
  if (!h.tienePares) return 'Sin pares';
  return `${h.pares.tipo} de ${VALUE_NAMES[h.pares.principal] ?? h.pares.principal}`;
}

function describeJuego(h) {
  return h.tieneJuego ? `Juego: ${h.juego}` : `Sin juego (${h.juego})`;
}

function renderHandInfo(h) {
  $('hand-info').replaceChildren(
    el('span', { className: `chip${h.tienePares ? ' good' : ''}`, textContent: describePares(h) }),
    el('span', { className: `chip${h.tieneJuego ? ' good' : ''}`, textContent: describeJuego(h) }),
  );
}

function renderActions() {
  const g = state.game;
  const box = $('actions');
  if (state.reveal || g.gameOver || g.current !== HUMAN) { box.replaceChildren(); return; }
  const keyOf = Object.fromEntries(Object.entries(KEYS).map(([k, v]) => [v, k.toUpperCase()]));
  box.replaceChildren(...g.legalActions().map((a) => {
    const label = a === Action.ENVIDO ? 'Envido (2)' : ACTION_LABELS[a];
    const b = el('button', { className: `btn${a === Action.ORDAGO ? ' danger' : ''}`, type: 'button' },
      [label, el('kbd', { textContent: keyOf[a] })]);
    b.addEventListener('click', () => humanPlay(a));
    return b;
  }));
}

function renderScores(scores) {
  for (const [id, s] of [['score-human', scores[HUMAN]], ['score-ai', scores[AI]]]) {
    const box = $(id);
    box.querySelector('strong').textContent = s;
    box.querySelector('.bar i').style.width = `${Math.min(100, (s / 30) * 100)}%`;
  }
}

function renderOverlay() {
  const r = state.reveal;
  const overlay = $('overlay');
  if (!r) { overlay.hidden = true; return; }
  overlay.hidden = false;

  let title = 'Fin de la mano';
  if (r.gameOver) title = r.winner === HUMAN ? '¡Has ganado la partida! 🏆' : 'La IA gana la partida';
  $('dialog-title').textContent = title;

  const handRow = (p) => {
    const h = r.hands[p];
    const cards = el('div', { className: 'cards' });
    renderCards(cards, h.cards, false);
    return el('div', { className: 'reveal-row' }, [
      el('div', {}, [el('span', { className: 'who', textContent: p === HUMAN ? 'Tú' : 'IA' }),
        el('small', { textContent: r.mano === p ? 'mano' : 'postre' })]),
      el('div', {}, [cards, el('small', { textContent: `${describePares(h)} · ${describeJuego(h)}` })]),
    ]);
  };

  const during = r.points.filter((p) => p.reason === 'no_quiero' || p.reason === 'solo');
  const tally = r.points.filter((p) => !during.includes(p));
  const list = (items) => el('ul', { className: 'tally' }, items.map((p) => el('li', {}, [
    el('span', { textContent: pointsReason(p) }),
    el('span', {
      className: `pts ${p.player === HUMAN ? 'h' : 'a'}`,
      textContent: p.reason === 'ordago' ? `${p.player === HUMAN ? 'Tú' : 'IA'} gana` : `+${p.points} ${p.player === HUMAN ? 'tú' : 'IA'}`,
    }),
  ])));

  const body = [el('div', { className: 'reveal' }, [handRow(AI), handRow(HUMAN)])];
  if (during.length) body.push(el('p', { className: 'tally-title', textContent: 'Durante la mano' }), list(during));
  if (tally.length) body.push(el('p', { className: 'tally-title', textContent: r.reason === 'ordago' ? 'Órdago' : 'Recuento' }), list(tally));
  if (!r.points.length) body.push(el('p', { className: 'muted', textContent: 'Nadie suma puntos en esta mano.' }));
  body.push(el('p', { className: 'muted', textContent: `Marcador: Tú ${r.scores[HUMAN]} – IA ${r.scores[AI]}` }));
  $('dialog-body').replaceChildren(...body);

  const btn = $('dialog-btn');
  btn.textContent = r.gameOver ? 'Jugar otra partida' : 'Siguiente mano';
  btn.focus({ preventScroll: true });
}

function continueAfterReveal() {
  if (!state.reveal) return;
  if (state.reveal.gameOver) { newGame(); return; }
  state.reveal = null;
  state.dealt = true;
  render();
  scheduleAi();
}

// ------------------------------------------------------------ panel IA y log
function showThinking(probs, legal, requested, played, lance) {
  $('think-lance').textContent = `en ${LANCE_LABELS[lance].toLowerCase()}`;
  $('probs').replaceChildren(...probs.map((p, a) => el('div', {
    className: `prob${a === requested ? ' chosen' : ''}${legal.includes(a) ? '' : ' illegal'}`,
  }, [
    el('span', { className: 'name', textContent: ACTION_LABELS[a] }),
    el('div', { className: 'track' }, el('div', { className: 'fill', style: `width:${(p * 100).toFixed(1)}%` })),
    el('span', { className: 'val', textContent: `${Math.round(p * 100)}%` }),
  ])));
  $('think-note').textContent = requested === played
    ? 'Probabilidades de la política (salida softmax). La IA elige la más alta.'
    : `Eligió «${ACTION_LABELS[requested]}», que no era legal: se sustituye por una legal al azar (igual que en el entrenamiento).`;
}

function showBubble(text) {
  const b = $('ai-bubble');
  b.textContent = text;
  b.classList.add('show');
  clearTimeout(showBubble.t);
  showBubble.t = setTimeout(() => b.classList.remove('show'), 1800);
}

function log(text, cls) {
  const list = $('log');
  list.append(el('li', { className: cls, textContent: text }));
  list.scrollTop = list.scrollHeight;
}

function setStatus(text) { $('status').textContent = text; }

// ------------------------------------------------------------ init
function init() {
  const select = $('model-select');
  for (const m of MODELS) select.append(el('option', { value: m.id, textContent: m.label, title: m.stats }));
  select.addEventListener('change', newGame);
  $('new-game').addEventListener('click', newGame);
  $('dialog-btn').addEventListener('click', continueAfterReveal);
  $('show-ai-cards').addEventListener('change', (e) => { state.showAiCards = e.target.checked; render(); });
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.ctrlKey || e.metaKey || e.altKey) return;
    if ((e.key === ' ' || e.key === 'Enter') && state.reveal) { e.preventDefault(); continueAfterReveal(); return; }
    const action = KEYS[e.key.toLowerCase()];
    if (action !== undefined) humanPlay(action);
  });
  newGame();
}

init();
