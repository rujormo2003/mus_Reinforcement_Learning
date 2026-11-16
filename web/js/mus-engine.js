/**
 * Port a JavaScript del motor de `mus_gym` (mus_env.py, mus_rules.py,
 * mus_entities.py, mus_observation.py).
 *
 * Mantiene exactamente las mismas reglas y la misma codificación de la
 * observación, porque la red neuronal exportada depende de ella. Además
 * emite eventos (acciones, puntos, fin de mano...) para la interfaz.
 */

export const Action = Object.freeze({ PASO: 0, ENVIDO: 1, ORDAGO: 2, QUIERO: 3, NO_QUIERO: 4 });
export const ACTION_LABELS = ['Paso', 'Envido', 'Órdago', 'Quiero', 'No quiero'];
export const LANCES = ['GRANDE', 'CHICA', 'PARES', 'JUEGO'];
export const LANCE_LABELS = ['Grande', 'Chica', 'Pares', 'Juego'];
const [GRANDE, CHICA, PARES, JUEGO] = [0, 1, 2, 3];
const START = 0, BET_MADE = 1;
const BET_RENUNCIADA = -1, BET_SIN_DISPUTA = -2;

// ---------------------------------------------------------------- baraja
const PALOS = ['Oros', 'Copas', 'Espadas', 'Bastos'];
const RANGOS = ['As', '2', '3', '4', '5', '6', '7', 'Sota', 'Caballo', 'Rey'];
const VALOR_JUEGO = { As: 1, 2: 1, 3: 10, 4: 4, 5: 5, 6: 6, 7: 7, Sota: 10, Caballo: 10, Rey: 10 };
const VALOR_CMP = { As: 1, 2: 1, 3: 10, 4: 4, 5: 5, 6: 6, 7: 7, Sota: 8, Caballo: 9, Rey: 10 };
const RANKING_JUEGO = [33, 34, 35, 36, 37, 40, 32, 31];
const RANK_PARES = { Ninguno: 0, Pares: 1, Medias: 2, Duples: 3 };

export function cardImage(card) {
  return `cards/${card.rango.toLowerCase()}_${card.palo.toLowerCase()}.webp`;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function newDeck() {
  const deck = [];
  for (const palo of PALOS) for (const rango of RANGOS) deck.push({ palo, rango });
  return shuffle(deck);
}

export class Hand {
  constructor(cards) {
    this.cards = cards;
    this.valores = cards.map((c) => VALOR_CMP[c.rango]).sort((a, b) => b - a);
    this.juego = cards.reduce((s, c) => s + VALOR_JUEGO[c.rango], 0);
    this.tieneJuego = this.juego >= 31;
    this.pares = Hand.paresInfo(this.valores);
    this.tienePares = this.pares.tipo !== 'Ninguno';
  }

  static paresInfo(valores) {
    const conteo = new Map();
    for (const v of valores) conteo.set(v, (conteo.get(v) || 0) + 1);
    const reps = [...conteo.values()];
    if (reps.includes(4)) {
      const v = conteo.keys().next().value;
      return { tipo: 'Duples', principal: v, secundario: v };
    }
    if (reps.includes(3)) {
      const v = [...conteo].find(([, c]) => c === 3)[0];
      return { tipo: 'Medias', principal: v, secundario: 0 };
    }
    const pares = [...conteo].filter(([, c]) => c === 2).map(([v]) => v);
    if (pares.length === 2) return { tipo: 'Duples', principal: Math.max(...pares), secundario: Math.min(...pares) };
    if (pares.length === 1) return { tipo: 'Pares', principal: pares[0], secundario: 0 };
    return { tipo: 'Ninguno', principal: 0, secundario: 0 };
  }
}

// ---------------------------------------------------------------- reglas
function cmpLists(a, b) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] > b[i]) return true;
    if (a[i] < b[i]) return false;
  }
  return null;
}
const winnerOf = (isWinner0, mano) => (isWinner0 === null ? mano : isWinner0 ? 0 : 1);
const juegoOrdenable = (p) => (p === 31 ? 100 : p === 32 ? 99 : p);

export const rules = {
  grande: (h0, h1, mano) => winnerOf(cmpLists(h0.valores, h1.valores), mano),
  chica: (h0, h1, mano) => {
    const r = cmpLists([...h0.valores].reverse(), [...h1.valores].reverse());
    return winnerOf(r === null ? null : !r, mano);
  },
  pares: (h0, h1, mano) => {
    const k = (h) => [RANK_PARES[h.pares.tipo], h.pares.principal, h.pares.secundario];
    return winnerOf(cmpLists(k(h0), k(h1)), mano);
  },
  juego: (h0, h1, mano) => {
    if (h0.tieneJuego !== h1.tieneJuego) return winnerOf(h0.tieneJuego, mano);
    const f = h0.tieneJuego ? juegoOrdenable : (x) => x;
    return winnerOf(cmpLists([f(h0.juego)], [f(h1.juego)]), mano);
  },
  paresPoints: (h) => ({ Pares: 1, Medias: 2, Duples: 3 }[h.pares.tipo] || 0),
  juegoPoints: (h) => (!h.tieneJuego ? 1 : h.juego === 31 ? 3 : 2),
};
const COMPARATORS = [rules.grande, rules.chica, rules.pares, rules.juego];

// ---------------------------------------------------------------- partida
export class MusGame {
  /**
   * Llama a `reset()` para empezar (devuelve los eventos del primer reparto).
   * `deal` (opcional) sustituye al reparto aleatorio: () => [cartas J0, cartas J1].
   */
  constructor(targetScore = 30, { deal = null } = {}) {
    this.target = targetScore;
    this.deal = deal;
  }

  reset({ mano = null } = {}) {
    this.scores = [0, 0];
    this.mano = mano ?? (Math.random() < 0.5 ? 0 : 1);
    this.gameOver = false;
    this.winner = null;
    this.events = [];
    this.handNumber = 0;
    this._startNewHand(true);
    return this.flushEvents();
  }

  flushEvents() {
    const ev = this.events;
    this.events = [];
    return ev;
  }

  legalActions() {
    if (this.lanceState === START) return [Action.PASO, Action.ENVIDO, Action.ORDAGO];
    if (this.bet >= this.target) return [Action.QUIERO, Action.NO_QUIERO];
    return [Action.QUIERO, Action.NO_QUIERO, Action.ORDAGO];
  }

  /** Igual que MusObservationBuilder.build_observation (vector de 8 floats). */
  observation(p) {
    const obs = new Float32Array(8);
    const h = this.hands[p];
    obs[0] = h.valores.filter((v) => v === 10).length / 4;
    obs[1] = h.valores.filter((v) => v === 1).length / 4;
    obs[2] = { Ninguno: 0, Pares: 0.33, Medias: 0.66, Duples: 1 }[h.pares.tipo];
    obs[3] = h.tieneJuego && RANKING_JUEGO.includes(h.juego)
      ? (RANKING_JUEGO.indexOf(h.juego) + 1) / RANKING_JUEGO.length : 0;
    obs[4] = this.lance === null ? -1 : [0, 0.33, 0.66, 1][this.lance];
    obs[5] = this.lanceState === START ? 0 : this.bet >= this.target ? 1 : 0.5;
    obs[6] = p === this.mano ? 1 : 0;
    obs[7] = (this.scores[p] - this.scores[1 - p]) / this.target;
    return obs;
  }

  step(actionValue) {
    if (this.gameOver) return [];
    const actor = this.current;
    const legal = this.legalActions();
    let action = actionValue;
    if (!legal.includes(action)) action = legal[Math.floor(Math.random() * legal.length)];

    this.events.push({ kind: 'action', player: actor, action, lance: this.lance, requested: actionValue });
    if (this.lanceState === START) this._actionStart(action);
    else this._actionResponse(action);
    return this.flushEvents();
  }

  // ---------------------------------------------------------- interno
  _addPoints(player, points, lance, reason) {
    if (points <= 0) return;
    this.scores[player] += points;
    this.events.push({ kind: 'points', player, points, lance, reason });
  }

  _startNewHand(first = false) {
    if (this.gameOver) return;
    if (!first) this.mano = 1 - this.mano;
    const deck = newDeck();
    const [c0, c1] = this.deal ? this.deal() : [deck.splice(0, 4), deck.splice(0, 4)];
    this.hands = [new Hand(c0), new Hand(c1)];
    this.pending = [0, 0, 0, 0];
    this.lanceStatus = ['pending', 'pending', 'pending', 'pending'];
    this.lance = null;
    this.handNumber += 1;
    this.events.push({ kind: 'new_hand', mano: this.mano, number: this.handNumber });
    this._moveToNextLance();
  }

  _setLance(lance) {
    this.lance = lance;
    this.current = this.mano;
    this.bet = 0;
    this.lastValidBet = 0;
    this.better = null;
    this.lanceState = START;
    this.lanceStatus[lance] = 'current';
    this.events.push({ kind: 'lance', lance });
  }

  _autoResolve(lance) {
    let has;
    if (lance === PARES) has = this.hands.map((h) => h.tienePares);
    else if (lance === JUEGO) has = this.hands.map((h) => h.tieneJuego);
    else return false;
    if (has[0] && has[1]) return false;

    if (has[0] !== has[1]) {
      const p = has[0] ? 0 : 1;
      this._addPoints(p, 1, lance, 'solo');
    }
    this.pending[lance] = BET_SIN_DISPUTA;
    this.lanceStatus[lance] = 'skipped';
    this.events.push({ kind: 'skip', lance, who: has[0] ? 0 : has[1] ? 1 : null });
    if (has[0] !== has[1]) this._checkGameOver();
    return true;
  }

  _moveToNextLance() {
    if (this.gameOver) return;
    if (this.lance !== null && this.lanceStatus[this.lance] === 'current') this.lanceStatus[this.lance] = 'done';
    for (let l = (this.lance === null ? -1 : this.lance) + 1; l < 4; l++) {
      if (!this._autoResolve(l)) {
        this._setLance(l);
        return;
      }
      if (this.gameOver) {
        this._emitHandEnd('limite');
        return;
      }
    }
    this._showdown();
    this._emitHandEnd('recuento');
    this._startNewHand();
  }

  _emitHandEnd(reason) {
    this.events.push({
      kind: 'hand_end', reason, hands: [this.hands[0], this.hands[1]],
      mano: this.mano, scores: [...this.scores], pending: [...this.pending],
    });
    if (this.gameOver) this.events.push({ kind: 'game_over', winner: this.winner, scores: [...this.scores] });
  }

  _actionStart(action) {
    const p = this.current;
    if (action === Action.PASO) {
      if (p === this.mano) this.current = 1 - p;
      else {
        this.pending[this.lance] = 0;
        this._moveToNextLance();
      }
    } else {
      this.lanceState = BET_MADE;
      this.bet = action === Action.ENVIDO ? 2 : this.target;
      this.lastValidBet = 0;
      this.better = p;
      this.current = 1 - p;
    }
  }

  _actionResponse(action) {
    const responder = this.current;
    const better = this.better;
    if (action === Action.QUIERO) {
      if (this.bet >= this.target) {
        const w = COMPARATORS[this.lance](this.hands[0], this.hands[1], this.mano);
        this.events.push({ kind: 'ordago', lance: this.lance, winner: w });
        this.scores[w] = this.target;
        this._checkGameOver();
        this._emitHandEnd('ordago');
      } else {
        this.pending[this.lance] = this.bet;
        this._moveToNextLance();
      }
    } else if (action === Action.NO_QUIERO) {
      const pts = this.lastValidBet > 0 ? this.lastValidBet : 1;
      this._addPoints(better, pts, this.lance, 'no_quiero');
      this.pending[this.lance] = BET_RENUNCIADA;
      this._checkGameOver();
      if (this.gameOver) this._emitHandEnd('limite');
      else this._moveToNextLance();
    } else if (action === Action.ORDAGO) {
      this.lastValidBet = this.bet;
      this.bet = this.target;
      this.better = responder;
      this.current = better;
    }
  }

  _showdown() {
    if (this.gameOver) return;
    const [h0, h1] = this.hands;
    for (let l = 0; l < 4; l++) {
      const bet = this.pending[l];
      if (bet === BET_SIN_DISPUTA) {
        if (l === PARES) {
          const g = h0.tienePares ? 0 : h1.tienePares ? 1 : null;
          if (g !== null) this._addPoints(g, rules.paresPoints(this.hands[g]), l, 'bonus');
        } else if (l === JUEGO) {
          const g = h0.tieneJuego ? 0 : h1.tieneJuego ? 1 : null;
          if (g !== null) this._addPoints(g, rules.juegoPoints(this.hands[g]), l, 'bonus');
        }
        continue;
      }
      if (bet === BET_RENUNCIADA) continue;

      const w = COMPARATORS[l](h0, h1, this.mano);
      if (bet > 0) this._addPoints(w, bet, l, 'envido');
      else if (l === GRANDE || l === CHICA) this._addPoints(w, 1, l, 'paso');
      if (l === PARES) this._addPoints(w, rules.paresPoints(this.hands[w]), l, 'bonus');
      else if (l === JUEGO) this._addPoints(w, rules.juegoPoints(this.hands[w]), l, 'bonus');
    }
    this._checkGameOver();
  }

  _checkGameOver() {
    const [s0, s1] = this.scores;
    if (s0 >= this.target || s1 >= this.target) {
      this.gameOver = true;
      this.winner = s0 !== s1 ? (s0 > s1 ? 0 : 1) : this.mano;
    }
  }
}
