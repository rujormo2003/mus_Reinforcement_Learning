"""
Evalúa uno o varios modelos entrenados contra los bots de referencia.

Ejemplos:
    python scripts/evaluate.py                                  # todos los modelos, 200 partidas
    python scripts/evaluate.py models/ppo_LinearDecay.zip -n 500
    python scripts/evaluate.py --opponents heuristic sheriff --plots

Rivales disponibles:
    random     acciones legales al azar
    heuristic  HeuristicBot (el rival de Mus-v1)
    sheriff    SheriffBot (acepta órdagos con manos medias)
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mus_gym import HeuristicBot, MusEnvConBots, SheriffBot  # noqa: E402


class RandomBot:
    def choose_action(self, observation):
        return random.choice(observation["legal_actions_enums"])


OPPONENTS = {
    "random": RandomBot,
    "heuristic": lambda: HeuristicBot(player_id=1),
    "sheriff": lambda: SheriffBot(player_id=1),
}


def play_games(model, opponent: str, n_games: int, seed: int = 0) -> list[dict]:
    random.seed(seed)
    env = MusEnvConBots(target_score=30, bot=OPPONENTS[opponent]())
    history = []
    for i in range(n_games):
        obs, _ = env.reset(seed=seed + i)
        lstm_state, episode_start = None, np.ones((1,), dtype=bool)
        done, steps = False, 0
        while not done:
            action, lstm_state = model.predict(obs, state=lstm_state,
                                               episode_start=episode_start, deterministic=True)
            episode_start[0] = False
            obs, _, done, _, _ = env.step(int(action))
            steps += 1
        history.append({"win": env.winner == 0, "steps": steps,
                        "score_ai": env.scores[0], "score_bot": env.scores[1]})
    return history


def save_plots(name: str, history: list[dict], out_dir: str):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(history)
    df["Resultado"] = df["win"].map({True: "VICTORIA", False: "DERROTA"})
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Resultado", y="steps", hue="Resultado", data=df, legend=False,
                palette={"VICTORIA": "green", "DERROTA": "red"})
    plt.title(f"Modelo: {name}\nDuración vs Resultado")
    plt.savefig(os.path.join(out_dir, f"{name}_boxplot.png"))
    plt.close()

    df["Rango_Duracion"] = pd.cut(df["steps"], bins=5)
    win_rate = df.groupby("Rango_Duracion", observed=False)["win"].mean() * 100
    plt.figure(figsize=(10, 6))
    win_rate.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.axhline(y=50, color="r", linestyle="--")
    plt.title(f"Modelo: {name}\n% Victorias según duración")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_barras.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="Rutas a .zip (por defecto, todos los de models/)")
    parser.add_argument("-n", "--games", type=int, default=200)
    parser.add_argument("--opponents", nargs="+", default=list(OPPONENTS), choices=list(OPPONENTS))
    parser.add_argument("--plots", action="store_true", help="Guarda gráficos en results/evaluation/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from sb3_contrib import RecurrentPPO

    paths = args.models or sorted(glob.glob(os.path.join(ROOT, "models", "*.zip")))
    out_dir = os.path.join(ROOT, "results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        model = RecurrentPPO.load(path, device="cpu")
        row = {"model": name}
        for opp in args.opponents:
            hist = play_games(model, opp, args.games, args.seed)
            row[opp] = 100 * np.mean([h["win"] for h in hist])
            row[f"{opp}_steps"] = np.mean([h["steps"] for h in hist])
            if args.plots and opp == "heuristic":
                save_plots(name, hist, out_dir)
        rows.append(row)
        print(f"{name:22s} " + "  ".join(f"{o}: {row[o]:5.1f}%" for o in args.opponents), flush=True)

    print(f"\n| Modelo | " + " | ".join(f"vs {o}" for o in args.opponents) + " |")
    print("|---|" + "---|" * len(args.opponents))
    for row in rows:
        print(f"| `{row['model']}` | " + " | ".join(f"{row[o]:.1f}%" for o in args.opponents) + " |")


if __name__ == "__main__":
    main()
