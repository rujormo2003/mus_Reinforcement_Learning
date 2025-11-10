# evaluete.py
import gymnasium as gym
import mus_gym  # noqa: F401
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
import time

model = RecurrentPPO.load("recurrent_ppo_mus_v0")
eval_env = gym.make("Mus-v0")


mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )

eval_env.close()

print("\nEvaluación completada.")
print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")
