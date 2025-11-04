import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Importa nuestra clase de entorno desde el otro archivo
from entorno_mus import MusEnvironment, MusAction 

# --- 1. Creación de Directorios ---
log_dir = "./logs/mus_ppo/"
model_dir = "./models/mus_ppo/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# --- 2. Instanciar el Entorno ---
print("Creando el entorno de Mus...")
env = MusEnvironment(target_score=30)

# (Opcional pero RECOMENDADO) Comprueba si nuestro entorno sigue las reglas de Gym
check_env(env)
print("¡Comprobación del entorno superada!")

# --- 3. Definición de Hiperparámetros ---
# (Estos son los que discutimos)
policy_kwargs = dict(net_arch=[128, 128]) # Dos capas ocultas de 128 neuronas

# --- 4. Crear el Modelo ---
# Esta es la estrategia "Deep PPO" con "Self-Play" implícito.
# El agente aprenderá a jugar en ambas posiciones (mano y postre)
# simplemente viendo las diferentes observaciones que recibe.

print("Creando el modelo PPO...")
model = PPO(
    policy="MlpPolicy",             # Red neuronal estándar (Multi-Layer Perceptron)
    env=env,
    verbose=1,                      # Imprime el progreso en la consola
    n_steps=2048,                   # Pasos antes de cada actualización
    batch_size=64,
    gamma=0.99,                     # Descuento para recompensas futuras
    ent_coef=0.05,                  # Coeficiente de entropía (exploración)
    learning_rate=3e-4,             # 0.0003
    policy_kwargs=policy_kwargs,
    tensorboard_log=log_dir         
)

# --- 5. Entrenar el Modelo ---
TIMESTEPS_TOTALES = 1000000

print(f"Empezando entrenamiento por {TIMESTEPS_TOTALES} timesteps...")
model.learn(
    total_timesteps=TIMESTEPS_TOTALES,
    tb_log_name="PPO_Mus_Run1",   
    progress_bar=True            
)

# --- 6. Guardar el Modelo ---
model_path = os.path.join(model_dir, "mus_agent_1M_2")
model.save(model_path)

print(f"--- Entrenamiento Completado ---")
print(f"Modelo guardado en: {model_path}")
print("Puedes ver los resultados ejecutando:")
print(f"tensorboard --logdir={log_dir}")