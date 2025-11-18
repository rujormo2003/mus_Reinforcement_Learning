# train.py
import gymnasium as gym
import mus_gym  # Importa tu entorno registrado
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
import time
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
MODELS_DIR = "models"
LOGS_DIR = "logs"
TIMESTEPS = 1_000_000 

# Asegurar directorios
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- 1. Generar ID único ---
# Ejemplo: "recurrent_ppo_mus_scratch_20231118_173000"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"recurrent_ppo_mus_scratch_{timestamp}"

print(f"--- Iniciando NUEVO entrenamiento desde cero: {run_name} ---")

# --- 2. Crear el entorno ---
env = gym.make("Mus-v1")

# --- 3. Definir el Modelo (DESDE CERO) ---
# Usamos "MlpLstmPolicy" porque el Mus requiere memoria (recordar qué pasó en lances anteriores)
model = RecurrentPPO(
    "MlpLstmPolicy",  
    env,
    verbose=1,
    tensorboard_log=LOGS_DIR,
    learning_rate=0.0001,  # <--- Más lento y seguro
    gamma=0.99,
    ent_coef=0.05,         # <--- Mucha más exploración obligatoria
    batch_size=64,
    n_steps=128
)

# (Opcional) Guardado periódico por seguridad
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=f"{MODELS_DIR}/{run_name}_checkpoints",
    name_prefix="mus_model"
)

print("Iniciando proceso de aprendizaje...")
start_time = time.time()

# --- 4. Entrenar ---
model.learn(
    total_timesteps=TIMESTEPS,
    tb_log_name=run_name,
    callback=checkpoint_callback
)

end_time = time.time()
print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

# --- 5. Guardar el modelo final ---
save_path = os.path.join(MODELS_DIR, run_name)
model.save(save_path)

print(f"Modelo final guardado en: {save_path}.zip")
env.close()