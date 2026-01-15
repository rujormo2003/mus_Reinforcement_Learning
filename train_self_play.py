import os
import numpy as np
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
import mus_gym  

MODELO_ANTERIOR = "models/ppo_lstm_mus_v3_vec_.zip" 
CARPETA_SELFPLAY = "models/selfplay_buffer_v2"
LOGS_DIR = "logs"
TIMESTEPS_TOTAL = 2_000_000  
UPDATE_FREQ = 50_000          
EVAL_EPISODES = 100           
WIN_THRESHOLD = 0.55          
NUM_ENVS = 4                  

os.makedirs(CARPETA_SELFPLAY, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

class ChampionLeagueCallback(BaseCallback):
    """
    Callback tipo 'Rey de la Colina'.
    Cada X pasos, pausa el entrenamiento y enfrenta al modelo actual (Aspirante)
    contra el modelo guardado en disco (Campeón).
    
    Si Aspirante gana > WIN_THRESHOLD, se convierte en el nuevo Campeón.
    """
    def __init__(self, update_freq, save_dir, eval_episodes=100, win_threshold=0.55, verbose=1):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_dir = save_dir
        self.eval_episodes = eval_episodes
        self.win_threshold = win_threshold
        self.last_update_step = 0
        
        self.champion_path = os.path.join(self.save_dir, "champion_latest.zip")
        self.candidate_path = os.path.join(self.save_dir, "candidate_temp.zip")

        self.eval_env = make_vec_env(
            "Mus-v3", 
            n_envs=1, 
            vec_env_cls=DummyVecEnv,
            env_kwargs={"opponent_path": self.champion_path}
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update_step >= self.update_freq:
            self.last_update_step = self.num_timesteps
            
            if self.verbose > 0:
                print(f"\n[League] Paso {self.num_timesteps}: ¡EL ASPIRANTE RETA AL CAMPEÓN!")

            self.model.save(self.candidate_path)
            
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.eval_episodes,
                deterministic=True 
            )
        
            
            print(f"[League] Resultado del duelo: Recompensa Media = {mean_reward:.3f} (std={std_reward:.3f})")
            
            is_new_champion = mean_reward > 0.05
            if is_new_champion:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_path = os.path.join(self.save_dir, f"champion_{self.num_timesteps}_{timestamp}.zip")
                
                self.model.save(self.champion_path)
                self.model.save(archive_path)      
                
                print(f"[League] NUEVO CAMPEÓN! El modelo actual ha derrotado al anterior.")
                
                self.training_env.env_method("update_opponent_model", self.champion_path)
                self.eval_env.env_method("update_opponent_model", self.champion_path)
                
            else:
                print(f"[League] El aspirante ha fallado. Se mantiene el Campeón actual.")
            
        return True

def main():
    print(f"--- Iniciando Entrenamiento League Self-Play (Mus-v3) ---")
    
    initial_champion_path = os.path.join(CARPETA_SELFPLAY, "champion_latest.zip")
    base_model = RecurrentPPO.load(MODELO_ANTERIOR)
    base_model.save(initial_champion_path)
    del base_model 
    env_kwargs = {"opponent_path": initial_champion_path}
    
    env = make_vec_env(
        "Mus-v3", 
        n_envs=NUM_ENVS, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )

    learner = RecurrentPPO.load(
        MODELO_ANTERIOR, 
        env=env,
        tensorboard_log=LOGS_DIR,
        device="auto",
        learning_rate=0.0002 
    )

    league_callback = ChampionLeagueCallback(
        update_freq=UPDATE_FREQ,
        save_dir=CARPETA_SELFPLAY,
        eval_episodes=EVAL_EPISODES,
        win_threshold=WIN_THRESHOLD
    )

    run_name = f"Mus_League_{datetime.now().strftime('%Y%m%d_%H%M')}"
    learner.learn(
        total_timesteps=TIMESTEPS_TOTAL, 
        callback=league_callback,
        tb_log_name=run_name,
        reset_num_timesteps=False 
    )

    learner.save(f"models/{run_name}_FINAL")
    env.close()
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()