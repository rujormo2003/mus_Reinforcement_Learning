# train_self_play.py
import os
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
import mus_gym  # Import necesario para registrar los envs

# --- CONFIGURACIÓN ---
MODELO_ANTERIOR = "models/ppo_lstm_mus_v3_vec_.zip" # Ruta de tu modelo entrenado previamente
CARPETA_SELFPLAY = "models/selfplay_buffer"
LOGS_DIR = "logs"
TIMESTEPS_TOTAL = 2_000_000  # Entrenar por largo tiempo
UPDATE_FREQ = 50_000         # Actualizar rival cada 50k pasos
NUM_ENVS = 2                 # Entornos paralelos

os.makedirs(CARPETA_SELFPLAY, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

class SelfPlayCallback(BaseCallback):
    """
    Callback personalizado para:
    1. Guardar el modelo actual cada X pasos.
    2. Actualizar el entorno para que use ese modelo como oponente.
    """
    def __init__(self, update_freq, save_dir, verbose=1):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.save_dir = save_dir
        self.last_update_step = 0

    def _on_step(self) -> bool:
        # num_timesteps se actualiza con cada paso en cada ambiente
        if self.num_timesteps - self.last_update_step >= self.update_freq:
            self.last_update_step = self.num_timesteps
            
            # 1. Definir ruta para el nuevo "champion"
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_opponent_path = os.path.join(self.save_dir, f"opponent_{self.num_timesteps}_{timestamp}.zip")
            latest_opponent_path = os.path.join(self.save_dir, "opponent_latest.zip")
            
            # 2. Guardar el modelo actual (el que está aprendiendo)
            self.model.save(new_opponent_path)
            self.model.save(latest_opponent_path) # Sobreescribir el latest para facilitar carga
            
            if self.verbose > 0:
                print(f"\n[Self-Play] ¡Nuevo campeón generado en paso {self.num_timesteps}!")
                print(f"[Self-Play] Actualizando oponentes en los entornos...")

            # 3. Notificar a los entornos que carguen el nuevo modelo
            # Usamos env_method para llamar a la función dentro de cada subproceso
            self.training_env.env_method("update_opponent_model", latest_opponent_path)
            
        return True

def main():
    print(f"--- Iniciando Entrenamiento Self-Play (Mus-v3) ---")
    print(f"Cargando modelo base: {MODELO_ANTERIOR}")

    # 1. Preparar el modelo inicial del oponente
    # Copiamos el modelo base a la carpeta buffer para que sea el primer rival
    initial_opponent_path = os.path.join(CARPETA_SELFPLAY, "opponent_initial.zip")
    base_model = RecurrentPPO.load(MODELO_ANTERIOR)
    base_model.save(initial_opponent_path)
    del base_model # Liberar memoria

    # 2. Crear entorno vectorizado Mus-v3
    # Nota: Pasamos 'opponent_path' en lugar de 'opponent_policy' objeto
    env_kwargs = {"opponent_path": initial_opponent_path}
    
    env = make_vec_env(
        "Mus-v3", 
        n_envs=NUM_ENVS, 
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )

    # 3. Cargar el Agente (Learner) desde el modelo previo
    # Importante: pasamos el 'env' nuevo para que se vincule correctamente
    learner = RecurrentPPO.load(
        MODELO_ANTERIOR, 
        env=env,
        tensorboard_log=LOGS_DIR,
        device="auto",
        # Puedes ajustar learning rate aquí si quieres hacer fine-tuning más suave
        learning_rate=0.0002 
    )

    # 4. Configurar Callback
    self_play_callback = SelfPlayCallback(
        update_freq=UPDATE_FREQ,
        save_dir=CARPETA_SELFPLAY
    )

    # 5. Entrenar
    run_name = f"Mus_SelfPlay_{datetime.now().strftime('%Y%m%d_%H%M')}"
    learner.learn(
        total_timesteps=TIMESTEPS_TOTAL, 
        callback=self_play_callback,
        tb_log_name=run_name,
        reset_num_timesteps=False # Mantiene el conteo histórico del modelo si existía
    )

    # 6. Guardar modelo final
    learner.save(f"models/{run_name}_FINAL")
    env.close()
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    main()