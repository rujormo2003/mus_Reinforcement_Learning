import os
import numpy as np
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
import mus_gym  # Import necesario para registrar los envs

# --- CONFIGURACIÓN ---
MODELO_ANTERIOR = "models/ppo_lstm_mus_v3_vec_.zip" 
CARPETA_SELFPLAY = "models/selfplay_buffer_v2"
LOGS_DIR = "logs"
TIMESTEPS_TOTAL = 2_000_000  
UPDATE_FREQ = 50_000          # Intentar retar al campeón cada 50k pasos
EVAL_EPISODES = 100           # Número de manos para decidir quién es mejor
WIN_THRESHOLD = 0.55          # El aspirante debe ganar el 55% de las veces para coronarse
NUM_ENVS = 4                  # Entornos paralelos para entrenamiento

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
        
        # Rutas de archivos
        self.champion_path = os.path.join(self.save_dir, "champion_latest.zip")
        self.candidate_path = os.path.join(self.save_dir, "candidate_temp.zip")

        # Inicializamos un entorno de evaluación separado.
        # Al inicio, el "champion_path" debe existir (se crea en main)
        # Usamos DummyVecEnv para evaluación (más simple que Subproc)
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

            # 1. Guardar el modelo actual (Aspirante) temporalmente
            self.model.save(self.candidate_path)
            
            # 2. Evaluar: Agent (Aspirante) vs Environment (Campeón cargado en mus-v3)
            # Nota: evaluate_policy devuelve (mean_reward, std_reward).
            # En Mus, si reward=1 es ganar y -1 es perder, un mean_reward > 0.1 significa que gana bastante.
            # Sin embargo, para ser precisos, idealmente tu entorno debería devolver 1 si gana la mano y 0/-1 si pierde.
            # Asumiremos que el reward refleja la victoria.
            
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.eval_episodes,
                deterministic=True # Usar la mejor jugada posible
            )
            
            # Estimación de Win Rate basada en reward (ajusta esto según tu función de recompensa)
            # Si tu reward es +/- diff de puntos, esto es solo una media de puntos.
            # Si tu reward es +1 ganar / -1 perder: WinRate = (mean + 1) / 2
            
            print(f"[League] Resultado del duelo: Recompensa Media = {mean_reward:.3f} (std={std_reward:.3f})")

            # Lógica de decisión: 
            # Asumiendo que recompensa positiva significa que el Aspirante fue mejor que el Campeón
            # Si usas reward puro de puntos, > 0 significa que ganó más puntos en promedio.
            
            is_new_champion = mean_reward > 0.05 # Un pequeño margen positivo (ajustar según tu lógica de reward)
            
            if is_new_champion:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_path = os.path.join(self.save_dir, f"champion_{self.num_timesteps}_{timestamp}.zip")
                
                # A. Promocionar a Campeón
                self.model.save(self.champion_path) # Sobreescribimos el 'latest'
                self.model.save(archive_path)       # Guardamos histórico
                
                print(f"[League] 👑 ¡TENEMOS NUEVO CAMPEÓN! El modelo actual ha derrotado al anterior.")
                
                # B. Actualizar los entornos de entrenamiento para usar el nuevo campeón
                self.training_env.env_method("update_opponent_model", self.champion_path)
                
                # C. Actualizar el entorno de evaluación también
                self.eval_env.env_method("update_opponent_model", self.champion_path)
                
            else:
                print(f"[League] ❌ El aspirante ha fallado. Se mantiene el Campeón actual.")
            
        return True

def main():
    print(f"--- Iniciando Entrenamiento League Self-Play (Mus-v3) ---")
    
    # 1. Preparar el PRIMER Campeón
    initial_champion_path = os.path.join(CARPETA_SELFPLAY, "champion_latest.zip")
    
    # Cargamos el modelo base y lo guardamos como el primer campeón
    base_model = RecurrentPPO.load(MODELO_ANTERIOR)
    base_model.save(initial_champion_path)
    del base_model 

    # 2. Crear entorno de entrenamiento
    # Todos los entornos empiezan jugando contra el campeón inicial
    env_kwargs = {"opponent_path": initial_champion_path}
    
    env = make_vec_env(
        "Mus-v3", 
        n_envs=NUM_ENVS, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )

    # 3. Cargar el Agente (Learner)
    learner = RecurrentPPO.load(
        MODELO_ANTERIOR, 
        env=env,
        tensorboard_log=LOGS_DIR,
        device="auto",
        learning_rate=0.0002 
    )

    # 4. Configurar el Callback de Liga
    league_callback = ChampionLeagueCallback(
        update_freq=UPDATE_FREQ,
        save_dir=CARPETA_SELFPLAY,
        eval_episodes=EVAL_EPISODES,
        win_threshold=WIN_THRESHOLD
    )

    # 5. Entrenar
    run_name = f"Mus_League_{datetime.now().strftime('%Y%m%d_%H%M')}"
    learner.learn(
        total_timesteps=TIMESTEPS_TOTAL, 
        callback=league_callback,
        tb_log_name=run_name,
        reset_num_timesteps=False 
    )

    # 6. Guardar final
    learner.save(f"models/{run_name}_FINAL")
    env.close()
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()