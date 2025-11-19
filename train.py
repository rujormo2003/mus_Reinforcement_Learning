# train.py - VERSIÓN VECTORIZADA Y AGRESIVA
import gymnasium as gym
import mus_gym  # Tu paquete
import torch
import os
import time
from datetime import datetime
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

# --- CONFIGURACIÓN ---
MODELS_DIR = "models"
LOGS_DIR = "logs"
TIMESTEPS = 1_000_000  # Ahora que vamos rápido, apuntamos alto (2 Millones)
NUM_ENVS = 8           # ¡8 Partidas simultáneas! (Ajusta según los núcleos de tu CPU)

def main():
    # Asegurar directorios
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Nombre único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_lstm_mus_v3_vec_"

    print(f"--- Iniciando entrenamiento VECTORIZADO: {run_name} ---")
    print(f"--- Procesos paralelos: {NUM_ENVS} ---")

    # 1. CREAR ENTORNOS VECTORIZADOS
    # Esto crea 8 copias independientes del juego.
    # Usamos SubprocVecEnv para que cada juego corra en un núcleo de CPU distinto.
    env = make_vec_env(
        "Mus-v1", 
        n_envs=NUM_ENVS, 
        vec_env_cls=SubprocVecEnv
    )

    # 2. DEFINIR MODELO (Más agresivo)
    model = RecurrentPPO(
        "MlpLstmPolicy",  
        env,
        verbose=1,
        tensorboard_log=LOGS_DIR,
        learning_rate=0.0002,   # Un pelín más rápido al principio
        gamma=0.995,            # Valorar más el futuro (ganar la partida al final)
        ent_coef=0.05,          # <--- ENTROPÍA SUBIDA (Antes 0.01/0.0) para forzar exploración
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=128,         # Lote más grande para estabilizar actualizaciones
        n_steps=256,            # Pasos por ambiente antes de actualizar (Total = 256 * 8 = 2048 pasos por update)
        policy_kwargs=dict(
            # Hacemos la red un poco más "inteligente" (más neuronas)
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            lstm_hidden_size=128
        )
    )

    # Guardado periódico
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // NUM_ENVS, # Ajustamos frecuencia porque los steps van x8
        save_path=f"{MODELS_DIR}/{run_name}_checkpoints",
        name_prefix="mus_model"
    )

    print("Iniciando proceso de aprendizaje acelerado...")
    start_time = time.time()

    # 3. ENTRENAR
    model.learn(
        total_timesteps=TIMESTEPS,
        tb_log_name=run_name,
        callback=checkpoint_callback
    )

    end_time = time.time()
    print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

    # 4. GUARDAR
    save_path = os.path.join(MODELS_DIR, run_name)
    model.save(save_path)
    print(f"Modelo guardado en: {save_path}.zip")
    
    env.close()

if __name__ == '__main__':
    # NECESARIO para Windows y Multiprocessing
    main()