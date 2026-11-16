"""
Fase 1 · Entrena un agente RecurrentPPO (LSTM) contra el HeuristicBot (Mus-v1).

Ejemplos:
    python scripts/train_vs_heuristic.py --timesteps 1000000
    python scripts/train_vs_heuristic.py --resume models/ppo_vs_heuristic.zip
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--resume", help="Continuar desde un .zip existente")
    parser.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    parser.add_argument("--logs-dir", default=os.path.join(ROOT, "logs"))
    args = parser.parse_args()

    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    from mus_gym import MusEnvConBots

    os.makedirs(args.models_dir, exist_ok=True)
    run_name = f"ppo_vs_heuristic_{datetime.now():%Y%m%d_%H%M%S}"
    env = make_vec_env(MusEnvConBots, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    if args.resume:
        model = RecurrentPPO.load(args.resume, env=env, tensorboard_log=args.logs_dir)
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=1, tensorboard_log=args.logs_dir,
            learning_rate=2e-4, gamma=0.995, ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5,
            batch_size=128, n_steps=256,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), lstm_hidden_size=128),
        )

    checkpoints = CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                                     save_path=os.path.join(args.models_dir, f"{run_name}_checkpoints"),
                                     name_prefix="mus_model")
    start = time.time()
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name, callback=checkpoints,
                reset_num_timesteps=not args.resume)
    print(f"Entrenamiento completado en {time.time() - start:.0f} s")

    save_path = os.path.join(args.models_dir, run_name)
    model.save(save_path)
    print(f"Modelo guardado en {save_path}.zip")
    env.close()


if __name__ == "__main__":
    main()
