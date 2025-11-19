# curriculum_train.py
import gymnasium as gym
import mus_gym  
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import time
from datetime import datetime
from mus_gym.mus_curriculum_env import MusEnvCurriculum 

# --- CONFIGURACIÓN GLOBAL ---
MODELS_DIR = "models"
LOGS_DIR = "logs"
TIMESTEPS_PER_RUN = 500_000 # Reducido para testing, sube a 1M para producción
NUM_ENVS = 8 # Número de ambientes paralelos internos
BASE_NAME = f"Curriculum_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 🛑 RUTA DEL PROFESOR (Debe ser el mismo que el modelo base del learner)
OPPONENT_MODEL_PATH = "models/ppo_lstm_mus_v3_vec_.zip" 


# --- DEFINICIÓN DE LAS 3 FASES ---
RUN_CONFIGS = [
    # FASE 1: Conservador - Asegurar que el fine-tuning no se vuelva loco
    {"id": 1, "name": f"{BASE_NAME}_R1_Conservative", "ent_coef": 0.01, "lr": 0.0001, "gamma": 0.99},
    
    # FASE 2: Balanceado - Buscar el punto dulce entre exploración y explotación
    {"id": 2, "name": f"{BASE_NAME}_R2_Balanced", "ent_coef": 0.04, "lr": 0.0002, "gamma": 0.995},
    
    # FASE 3: Agresivo - Forzar el riesgo y la innovación (alta entropía)
    {"id": 3, "name": f"{BASE_NAME}_R3_Aggressive", "ent_coef": 0.08, "lr": 0.0003, "gamma": 0.999},
]


def make_env_function(opponent_policy):
    """ Función que será llamada por make_vec_env para crear cada subproceso. """
    # Retorna la función que crea el entorno con el objeto del profesor
    return lambda: MusEnvCurriculum(opponent_policy=opponent_policy)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # 1. CARGAR EL MODELO PROFESOR UNA ÚNICA VEZ
    print(f"Cargando política base del Profesor desde {OPPONENT_MODEL_PATH}...")
    try:
        opponent_policy_object = RecurrentPPO.load(OPPONENT_MODEL_PATH)
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo oponente desde {OPPONENT_MODEL_PATH}. {e}")
        return

    # 2. EJECUCIÓN SECUENCIAL DE LAS FASES
    for config in RUN_CONFIGS:
        print("\n" + "="*50)
        print(f"[{config['name']}] INICIANDO FASE {config['id']}")
        print("="*50)
        
        # A. Crear Entorno Vectorizado (Profesor congelado ya cargado)
        env_func = make_env_function(opponent_policy_object) 
        env = make_vec_env(
            env_func, 
            n_envs=NUM_ENVS, 
            vec_env_cls=SubprocVecEnv
        )

        # B. CARGAR MODELO LEARNER (FINE-TUNING)
        # El Learner es una COPIA del Profesor que será entrenada
        model = RecurrentPPO.load(
            OPPONENT_MODEL_PATH,
            env=env,
            verbose=1,
            tensorboard_log=LOGS_DIR,
            # Sobreescribir HPs para la fase actual:
            learning_rate=config['lr'],
            ent_coef=config['ent_coef'],
            gamma=config['gamma'],
        )

        # C. ENTRENAR
        print(f"Entropia: {config['ent_coef']} | Learning Rate: {config['lr']}")
        model.learn(
            total_timesteps=TIMESTEPS_PER_RUN,
            tb_log_name=config['name'],
            reset_num_timesteps=True
        )

        # D. GUARDAR RESULTADO
        save_path = os.path.join(MODELS_DIR, config['name'])
        model.save(save_path)
        print(f"✅ FASE {config['id']} FINALIZADA. Guardado en {save_path}.zip")
        
        env.close()

    print("\n\n--- TODAS LAS FASES DE CURRÍCULUM HAN CONCLUIDO. ---")

if __name__ == '__main__':
    main()