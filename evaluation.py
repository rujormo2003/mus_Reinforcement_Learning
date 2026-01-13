import sys
import os
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

# --- BLOQUE DE IMPORTACIÓN ---
try:
    from mus_gym.mus_env_v2 import MusEnvConBots
    print("Entorno cargado: MusEnvConBots (IA vs HeuristicBot)")
except ImportError:
    try:
        from mus_env_v2 import MusEnvConBots
        print("Entorno local cargado: MusEnvConBots")
    except ImportError:
        print("Error: No se encuentra 'mus_env_v2.py'.")
        sys.exit(1)

def run_evaluation():
    # --- CONFIGURACIÓN ---
    MODEL_PATH = "models/sheriff_colab/FINAL_MUS_MODEL.zip"
    N_EPISODES = 100  # Número de partidas para sacar la media
    TARGET_SCORE = 30 # Piedras para ganar
    
    # 1. Cargar Entorno y Modelo
    env = MusEnvConBots(target_score=TARGET_SCORE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No existe el modelo en {MODEL_PATH}")
        return

    print(f"Cargando modelo... {MODEL_PATH}")
    try:
        model = RecurrentPPO.load(MODEL_PATH)
        is_recurrent = True
    except:
        print("Aviso: Cargando como PPO standard (no recurrente).")
        model = PPO.load(MODEL_PATH)
        is_recurrent = False

    print(f"\nIniciando evaluación de {N_EPISODES} partidas...")
    print(f"Partidas a {TARGET_SCORE} piedras.")
    print("IA (Tú) vs HeuristicBot (Experto)...")
    print("-" * 50)

    # 2. Bucle de Evaluación
    wins_ia = 0
    wins_bot = 0
    total_score_ia = 0
    total_score_bot = 0

    for i in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        
        # Resetear estados LSTM
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts,
                    deterministic=False
                )
            else:
                action, _ = model.predict(obs, deterministic=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if is_recurrent:
                episode_starts[0] = done

        # Fin de la partida
        score_ia = env.scores[0]
        score_bot = env.scores[1]
        
        # Acumulamos puntos totales para la media
        total_score_ia += score_ia
        total_score_bot += score_bot
        
        if score_ia >= TARGET_SCORE and score_ia > score_bot:
            wins_ia += 1
            res = "VICTORIA"
        else:
            wins_bot += 1
            res = "DERROTA"
            
        # Log ligero cada 10 partidas
        if (i + 1) % 10 == 0:
            print(f"Partida {i+1}/{N_EPISODES} -> {res} ({score_ia} - {score_bot})")

    # 3. Estadísticas Finales
    win_rate = (wins_ia / N_EPISODES) * 100
    avg_ia = total_score_ia / N_EPISODES
    avg_bot = total_score_bot / N_EPISODES

    print("\n" + "="*50)
    print(f" RESULTADOS FINALES ({N_EPISODES} partidas)")
    print("="*50)
    print(f"IA (Tú) Ganadas   : {wins_ia}")
    print(f"Bot (Rival) Ganadas: {wins_bot}")
    print("-" * 50)
    print(f"PORCENTAJE DE VICTORIAS: {win_rate:.2f}%")
    print("-" * 50)
    # Aquí está el formato solicitado: Media - Media
    print(f"PROMEDIO DE RESULTADOS (TU - BOT): {avg_ia:.2f} - {avg_bot:.2f}")
    print("="*50)

    if win_rate > 55:
        print("✅ CONCLUSIÓN: Tu modelo es SUPERIOR al Bot.")
    elif win_rate < 45:
        print("❌ CONCLUSIÓN: Tu modelo necesita mejorar.")
    else:
        print("⚠️ CONCLUSIÓN: Duelo reñido (Nivel similar).")

if __name__ == "__main__":
    run_evaluation()