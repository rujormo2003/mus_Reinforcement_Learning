/**
 * Inferencia de la política RecurrentPPO (sb3-contrib) en JavaScript puro.
 *
 *   obs(8) ─► LSTM actor ─► [Linear + tanh] × N ─► action_net ─► logits(5)
 *
 * Los pesos se generan con `scripts/export_web_policy.py`. El manifiesto
 * incluye casos de prueba calculados con PyTorch que `selfTest()` verifica.
 */

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function matVec(W, rows, cols, x, out, accumulate = false) {
  for (let r = 0; r < rows; r++) {
    let s = accumulate ? out[r] : 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) s += W[base + c] * x[c];
    out[r] = s;
  }
  return out;
}

export function softmax(logits) {
  const m = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export class LstmPolicy {
  constructor(manifest, buffer) {
    this.meta = manifest;
    const data = new Float32Array(buffer);
    const t = (key) => {
      const { shape, offset } = manifest.tensors[key];
      const size = shape.reduce((a, b) => a * b, 1);
      return { W: data.subarray(offset, offset + size), shape };
    };
    this.H = manifest.hidden_size;
    this.inSize = manifest.obs_size;
    this.wih = t('lstm.weight_ih').W;
    this.whh = t('lstm.weight_hh').W;
    this.bias = t('lstm.bias').W;
    this.mlp = [];
    for (let i = 0; i < manifest.mlp_layers; i++) {
      const w = t(`mlp.${i}.weight`);
      this.mlp.push({ W: w.W, out: w.shape[0], in: w.shape[1], b: t(`mlp.${i}.bias`).W });
    }
    const a = t('action.weight');
    this.action = { W: a.W, out: a.shape[0], in: a.shape[1], b: t('action.bias').W };
    this.gates = new Float32Array(4 * this.H);
    this.reset();
  }

  static async load(baseUrl) {
    const [manifest, buffer] = await Promise.all([
      fetch(`${baseUrl}.json`).then((r) => r.json()),
      fetch(`${baseUrl}.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new LstmPolicy(manifest, buffer);
  }

  /** Estado LSTM a cero (equivale a `episode_start=True`). */
  reset() {
    this.h = new Float32Array(this.H);
    this.c = new Float32Array(this.H);
  }

  /** Un paso de la red. Actualiza el estado LSTM y devuelve los logits. */
  forward(obs, episodeStart = false) {
    if (episodeStart) this.reset();
    const H = this.H;
    const g = this.gates;
    matVec(this.wih, 4 * H, this.inSize, obs, g);
    matVec(this.whh, 4 * H, H, this.h, g, true);
    const h = new Float32Array(H);
    const c = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      // Orden de puertas de PyTorch: input, forget, cell (g), output.
      const i = sigmoid(g[j] + this.bias[j]);
      const f = sigmoid(g[H + j] + this.bias[H + j]);
      const gg = Math.tanh(g[2 * H + j] + this.bias[2 * H + j]);
      const o = sigmoid(g[3 * H + j] + this.bias[3 * H + j]);
      c[j] = f * this.c[j] + i * gg;
      h[j] = o * Math.tanh(c[j]);
    }
    this.h = h;
    this.c = c;

    let x = h;
    for (const layer of this.mlp) {
      const y = matVec(layer.W, layer.out, layer.in, x, new Float32Array(layer.out));
      for (let k = 0; k < layer.out; k++) y[k] = Math.tanh(y[k] + layer.b[k]);
      x = y;
    }
    const logits = matVec(this.action.W, this.action.out, this.action.in, x, new Float32Array(this.action.out));
    for (let k = 0; k < logits.length; k++) logits[k] += this.action.b[k];
    return logits;
  }

  /** Elige acción. `deterministic` = argmax (como en la evaluación). */
  act(obs, { episodeStart = false, deterministic = true } = {}) {
    const logits = this.forward(obs, episodeStart);
    const probs = softmax(logits);
    let action;
    if (deterministic) {
      action = probs.indexOf(Math.max(...probs));
    } else {
      let r = Math.random();
      action = probs.findIndex((p) => (r -= p) <= 0);
      if (action < 0) action = probs.length - 1;
    }
    return { action, probs, logits };
  }

  /** Compara con los logits calculados por PyTorch al exportar. */
  selfTest(tolerance = 1e-3) {
    const saved = [this.h, this.c];
    this.reset();
    let maxErr = 0;
    for (const chk of this.meta.checks || []) {
      const out = this.forward(Float32Array.from(chk.obs), chk.episode_start);
      chk.logits.forEach((v, k) => { maxErr = Math.max(maxErr, Math.abs(v - out[k])); });
    }
    [this.h, this.c] = saved;
    return { ok: maxErr < tolerance, maxErr };
  }
}
