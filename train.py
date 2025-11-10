# train.py
import gymnasium as gym
import mus_gym  # noqa: F401
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
import time

# --- 2. Crear el entorno ---
# SB3 gestiona la vectorización por ti
env = gym.make("Mus-v0")
print(f"Espacio de Observación: {env.observation_space.shape}")
print(f"Espacio de Acción: {env.action_space.n}")

# --- 3. Definir y Entrenar el Modelo ---
# "MlpPolicy" es una política de red neuronal estándar
model = RecurrentPPO(
    "MlpLstmPolicy", 
    env, 
    verbose=1, # Imprime el progreso del entrenamiento
    tensorboard_log="./logs/"
)



print("Iniciando entrenamiento...")
start_time = time.time()

# ¡Entrenar al agente!
model.learn(
    total_timesteps=200_000,
    tb_log_name="recurrent_ppo_mus_v0"
)

end_time = time.time()
print(f"Entrenamiento completado en {end_time - start_time:.2f} segundos.")

# --- 4. Guardar el modelo ---
model_path = "recurrent_ppo_mus_v0"
model.save(model_path)
print(f"Modelo guardado en {model_path}.zip")

env.close()


# Para ver el progreso, ejecuta en otra terminal:
# tensorboard --logdir ./mus_tensorboard/