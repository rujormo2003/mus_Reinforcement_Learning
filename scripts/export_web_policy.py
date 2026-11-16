"""
Exporta la parte "actor" de un modelo RecurrentPPO para la demo web.

La demo (web/) ejecuta la red en JavaScript puro, sin PyTorch ni servidor:
    obs(8) -> LSTM actor -> MLP (tanh) -> action_net -> logits(5)

Genera en web/models/:
    <nombre>.bin   pesos float32 concatenados (little-endian)
    <nombre>.json  manifiesto: forma y offset de cada tensor + casos de prueba

Ejemplos:
    python scripts/export_web_policy.py models/ppo_LinearDecay.zip
    python scripts/export_web_policy.py models/*.zip
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "web", "models")


def export(path: str, out_dir: str, n_checks: int = 12) -> str:
    import torch
    from sb3_contrib import RecurrentPPO

    model = RecurrentPPO.load(path, device="cpu")
    policy = model.policy
    if policy.lstm_actor.num_layers != 1:
        raise ValueError("Solo se soportan LSTM de una capa")

    tensors: dict[str, torch.Tensor] = {
        "lstm.weight_ih": policy.lstm_actor.weight_ih_l0,
        "lstm.weight_hh": policy.lstm_actor.weight_hh_l0,
        "lstm.bias": policy.lstm_actor.bias_ih_l0 + policy.lstm_actor.bias_hh_l0,
    }
    mlp_layers = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    for i, layer in enumerate(mlp_layers):
        tensors[f"mlp.{i}.weight"] = layer.weight
        tensors[f"mlp.{i}.bias"] = layer.bias
    tensors["action.weight"] = policy.action_net.weight
    tensors["action.bias"] = policy.action_net.bias

    name = os.path.splitext(os.path.basename(path))[0]
    manifest = {
        "name": name,
        "num_timesteps": int(model.num_timesteps),
        "obs_size": int(policy.observation_space.shape[0]),
        "hidden_size": int(policy.lstm_actor.hidden_size),
        "mlp_layers": len(mlp_layers),
        "activation": "tanh",
        "tensors": {},
    }
    blobs, offset = [], 0
    for key, t in tensors.items():
        arr = t.detach().cpu().numpy().astype("<f4")
        manifest["tensors"][key] = {"shape": list(arr.shape), "offset": offset}
        blobs.append(arr.tobytes())
        offset += arr.size

    # Casos de prueba: la versión JS debe reproducir estos logits.
    rng = np.random.default_rng(0)
    obs_seq = rng.uniform(-0.2, 1.0, size=(n_checks, manifest["obs_size"])).astype(np.float32)
    starts = np.zeros(n_checks, dtype=bool)
    starts[0] = starts[n_checks // 2] = True
    checks, state = [], None
    for obs, start in zip(obs_seq, starts):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[None])
            if state is None:
                h = torch.zeros(1, 1, manifest["hidden_size"])
                state = (h, h.clone())
            features = policy.extract_features(obs_t, policy.pi_features_extractor)
            latent, state = policy._process_sequence(features, state, torch.as_tensor([float(start)]),
                                                     policy.lstm_actor)
            logits = policy.action_net(policy.mlp_extractor.forward_actor(latent))
        checks.append({"obs": obs.tolist(), "episode_start": bool(start),
                       "logits": logits[0].tolist()})
    manifest["checks"] = checks

    # Coherencia con model.predict() de SB3 (determinista).
    st, ep = None, np.ones((1,), dtype=bool)
    for c in checks:
        ep[0] = c["episode_start"]
        a, st = model.predict(np.array(c["obs"], dtype=np.float32), state=st, episode_start=ep,
                              deterministic=True)
        assert int(a) == int(np.argmax(c["logits"])), "La exportación no coincide con model.predict()"

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.bin"), "wb") as f:
        f.write(b"".join(blobs))
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    size_kb = offset * 4 / 1024
    print(f"{name}: {offset:,} parámetros ({size_kb:.0f} KB) -> {out_dir}")
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()
    sys.path.insert(0, ROOT)
    for path in args.models:
        export(path, args.out)


if __name__ == "__main__":
    main()
