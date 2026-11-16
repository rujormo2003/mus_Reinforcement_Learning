"""
Fase 2 · Self-play tipo "liga" (rey de la colina) con Mus-v3.

El agente entrena contra una copia congelada de sí mismo (el campeón). Cada
`--update-freq` pasos se enfrentan aspirante y campeón; si el aspirante supera
`--threshold` de recompensa media, pasa a ser el nuevo campeón y todos los
entornos actualizan su rival.

Ejemplo:
    python scripts/train_self_play.py --init models/ppo_vs_heuristic.zip --timesteps 2000000
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402


class ChampionLeagueCallback(BaseCallback):
    def __init__(self, update_freq: int, save_dir: str, eval_episodes: int = 100,
                 threshold: float = 0.05, verbose: int = 1):
        super().__init__(verbose)
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv

        from mus_gym import MusEnvSelfPlay

        self.update_freq = update_freq
        self.save_dir = save_dir
        self.eval_episodes = eval_episodes
        self.threshold = threshold
        self.last_update_step = 0
        self.champion_path = os.path.join(save_dir, "champion_latest.zip")
        self.eval_env = make_vec_env(MusEnvSelfPlay, n_envs=1, vec_env_cls=DummyVecEnv,
                                     env_kwargs={"opponent_path": self.champion_path})

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update_step < self.update_freq:
            return True
        self.last_update_step = self.num_timesteps

        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(self.model, self.eval_env,
                                                  n_eval_episodes=self.eval_episodes, deterministic=True)
        print(f"[Liga] paso {self.num_timesteps}: recompensa media aspirante = {mean_reward:.3f} ± {std_reward:.3f}")

        if mean_reward > self.threshold:
            self.model.save(self.champion_path)
            self.model.save(os.path.join(self.save_dir, f"champion_{self.num_timesteps}.zip"))
            self.training_env.env_method("update_opponent_model", self.champion_path)
            self.eval_env.env_method("update_opponent_model", self.champion_path)
            print("[Liga] ¡Nuevo campeón!")
        else:
            print("[Liga] Se mantiene el campeón actual.")
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", required=True, help="Modelo inicial (.zip) para aspirante y campeón")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--update-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--logs-dir", default=os.path.join(ROOT, "logs"))
    args = parser.parse_args()

    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from mus_gym import MusEnvSelfPlay

    run_name = f"selfplay_league_{datetime.now():%Y%m%d_%H%M}"
    save_dir = os.path.join(ROOT, "models", run_name)
    os.makedirs(save_dir, exist_ok=True)

    champion_path = os.path.join(save_dir, "champion_latest.zip")
    RecurrentPPO.load(args.init).save(champion_path)

    env = make_vec_env(MusEnvSelfPlay, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv,
                       env_kwargs={"opponent_path": champion_path})
    learner = RecurrentPPO.load(args.init, env=env, tensorboard_log=args.logs_dir, learning_rate=2e-4)

    callback = ChampionLeagueCallback(args.update_freq, save_dir, args.eval_episodes, args.threshold)
    learner.learn(total_timesteps=args.timesteps, callback=callback, tb_log_name=run_name,
                  reset_num_timesteps=False)
    learner.save(os.path.join(save_dir, "final"))
    env.close()
    print(f"Entrenamiento finalizado. Modelos en {save_dir}")


if __name__ == "__main__":
    main()
