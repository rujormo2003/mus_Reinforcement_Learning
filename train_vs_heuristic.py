import gymnasium as gym
import mus_gym
import torch
import os
import time
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

MODELS_DIR = "models"
LOGS_DIR = "logs"
TIMESTEPS = 1_000_000  
NUM_ENVS = 8           

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_lstm_mus_v3_vec_more_training_{timestamp}"

    print(f"--- Iniciando entrenamiento VECTORIZADO: {run_name} ---")
    print(f"--- Procesos paralelos: {NUM_ENVS} ---")

    env = make_vec_env(
        "Mus-v1", 
        n_envs=NUM_ENVS, 
        vec_env_cls=SubprocVecEnv
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",  
        env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        learning_rate=0.0002,   
        gamma=0.995,            
        ent_coef=0.05,          
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=128,        
        n_steps=256,            
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            lstm_hidden_size=128
        )
    )
    model = model.load("models/ppo_lstm_mus_v3_vec_.zip", env=env)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // NUM_ENVS, 
        save_path=f"{MODELS_DIR}/{run_name}_checkpoints",
        name_prefix="mus_model"
    )

    print("Iniciando proceso de aprendizaje acelerado...")
    start_time = time.time()

    model.learn(
        total_timesteps=TIMESTEPS,
        tb_log_name=run_name,
        callback=checkpoint_callback
    )

    end_time = time.time()
    print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

    save_path = os.path.join(MODELS_DIR, run_name)
    model.save(save_path)
    print(f"Modelo guardado en: {save_path}.zip")
    
    env.close()

if __name__ == '__main__':
    main()