# check_env.py
import gymnasium as gym
import mus_gym  # noqa: F401
from stable_baselines3.common.env_checker import check_env


print("Creando entorno 'Mus-v0'...")
env = gym.make("Mus-v0")

# --- 2. Ejecutar el chequeador de SB3 ---
print("Ejecutando chequeo de compatibilidad de SB3...")
try:
    check_env(env)
    print("¡ÉXITO! El entorno es compatible con SB3.")
except Exception as e:
    print("¡FALLO! El entorno tiene problemas:")
    print(e)

env.close()