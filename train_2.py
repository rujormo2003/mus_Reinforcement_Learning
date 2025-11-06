# --- Guardar como: entrenar_v2.py ---
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# ¡¡Importamos el NUEVO entorno!!
from entorno_v2 import MusEnvConBots

# --- 1. Creación de Directorios ---
log_dir = "./logs/mus_ppo_v3/" # Nuevo log dir
model_dir = "./models/mus_ppo_v3/" # Nuevo model dir
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# --- 2. Instanciar el Entorno V2 ---
print("Creando el entorno de Mus V2 (con Bots)...")
env = MusEnvConBots(target_score=30)
# check_env(env) # ¡No lo uses! Falla porque nuestro step() es "mágico"
print("Entorno V2 creado.")

# --- 3. Crear el Modelo ---
# Podemos empezar de cero O cargar el modelo anterior
# Empecemos de cero para ver si aprende
print("Creando el modelo PPO...")
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    ent_coef=0.01, # Volvemos a una entropía baja
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[128, 128]),
    tensorboard_log=log_dir
)
# --- Opcional: Cargar el modelo V1 para re-entrenarlo ---
# print("Cargando modelo V1 para re-entrenar...")
# model = PPO.load("./models/mus_ppo/mus_agent_1M", env=env)
# model.set_logger(new_logger) # (Se necesita un logger nuevo)

# --- 4. Entrenar el Modelo ---
TIMESTEPS_TOTALES = 1_000_000
print(f"Empezando entrenamiento V2 por {TIMESTEPS_TOTALES} timesteps...")
model.learn(
    total_timesteps=TIMESTEPS_TOTALES,
    tb_log_name="PPO_Mus_V2_vs_Bots",
    progress_bar=True
)

# --- 5. Guardar el Modelo V2 ---
model_path = os.path.join(model_dir, "mus_agent_v3_norm_1M")
model.save(model_path)
print(f"Modelo V2 guardado en: {model_path}")