import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

# --- 1. CONFIGURACIÓN DE RUTAS ROBUSTA ---
# Esto asegura que funcione sin importar desde qué carpeta ejecutes la terminal
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Carpeta: evaluation_and_game
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)             # Carpeta: Raíz del proyecto

# Añadimos la raíz al path para poder importar mus_gym
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- BLOQUE DE IMPORTACIÓN DE ENTORNO ---
try:
    from mus_gym.mus_env_v2 import MusEnvConBots
    print("✅ Entorno cargado correctamente desde la raíz.")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print(f"Buscando en: {PROJECT_ROOT}")
    sys.exit(1)

def run_evaluation():
    # --- CONFIGURACIÓN ---
    N_EPISODES = 100 
    TARGET_SCORE = 30 
    
    # --- SELECCIÓN DEL MODELO (Basado en tu imagen) ---
    # Aquí construimos la ruta exacta usando la raíz del proyecto.
    # He puesto uno de los que se ven en la foto: 'models/sheriff_colab/Mus_League_...'
    
    MODEL_REL_PATH = os.path.join("models", "ppo_LinearDecay.zip")
    MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_REL_PATH)
    
    # Si quieres probar otro, comenta la línea de arriba y descomenta/edita esta:
    # MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "selfplay_buffer_v2", "champion_latest.zip")

    # --- NOMBRES DE ARCHIVO DE SALIDA ---
    model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
    
    # Guardamos los resultados en la misma carpeta donde está este script (evaluation_and_game)
    plot_boxplot_name = os.path.join(SCRIPT_DIR, f"{model_name}_boxplot.png")
    plot_barras_name = os.path.join(SCRIPT_DIR, f"{model_name}_barras.png")

    # 1. Cargar Entorno
    env = MusEnvConBots(target_score=TARGET_SCORE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: No se encuentra el archivo del modelo.")
        print(f"Buscando en: {MODEL_PATH}")
        print("Verifica que el nombre del archivo .zip en el código coincida con el de la carpeta.")
        return

    print(f"\nCargando modelo... {model_name}")
    try:
        model = RecurrentPPO.load(MODEL_PATH)
        is_recurrent = True
        print("Modelo Recurrente (LSTM) detectado.")
    except:
        print("Aviso: Cargando como PPO standard (no recurrente).")
        model = PPO.load(MODEL_PATH)
        is_recurrent = False

    print(f"\nIniciando evaluación de {N_EPISODES} partidas...")
    print("-" * 50)

    history = []
    wins_ia = 0
    wins_bot = 0
    total_score_ia = 0
    total_score_bot = 0
    total_steps_all_games = 0

    # 2. Bucle de juego
    for i in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        current_episode_steps = 0

        while not done:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_starts,
                    deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            current_episode_steps += 1
            done = terminated or truncated
            
            if is_recurrent:
                episode_starts[0] = done

        total_steps_all_games += current_episode_steps
        score_ia = env.scores[0]
        score_bot = env.scores[1]
        
        if score_ia >= TARGET_SCORE and score_ia > score_bot:
            wins_ia += 1
            res_str = "VICTORIA"
            winner_code = 1 
        else:
            wins_bot += 1
            res_str = "DERROTA"
            winner_code = 0 

        history.append({
            'Partida': i + 1,
            'Resultado': res_str,
            'Ganador_Binario': winner_code,
            'Duracion_Pasos': current_episode_steps,
            'Puntos_IA': score_ia,
            'Puntos_Bot': score_bot
        })
            
        if (i + 1) % 10 == 0:
            print(f"Partida {i+1}/{N_EPISODES} -> {res_str} ({score_ia}-{score_bot}) | Pasos: {current_episode_steps}")

    # 3. Estadísticas
    win_rate = (wins_ia / N_EPISODES) * 100
    avg_steps = total_steps_all_games / N_EPISODES

    print("\n" + "="*50)
    print(f" RESULTADOS: {model_name}")
    print("="*50)
    print(f"IA Ganadas: {wins_ia} | Bot Ganadas: {wins_bot}")
    print(f"WIN RATE: {win_rate:.2f}%")
    print(f"DURACIÓN MEDIA: {avg_steps:.2f} pasos")
    print("="*50)

    # 4. Gráficos
    print("\nGenerando gráficos...")
    df = pd.DataFrame(history)
    sns.set_theme(style="whitegrid")
    
    # Gráfico 1
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Resultado', y='Duracion_Pasos', data=df, palette={'VICTORIA': 'green', 'DERROTA': 'red'})
    plt.title(f'Modelo: {model_name}\nDuración vs Resultado')
    plt.savefig(plot_boxplot_name)
    print(f"Gráfico guardado: {plot_boxplot_name}")
    plt.close() # Importante cerrar para liberar memoria
    
    # Gráfico 2
    try:
        df['Rango_Duracion'] = pd.cut(df['Duracion_Pasos'], bins=5) 
        win_rate_by_len = df.groupby('Rango_Duracion', observed=False)['Ganador_Binario'].mean() * 100
        
        plt.figure(figsize=(10, 6))
        win_rate_by_len.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Modelo: {model_name}\n% Victorias según duración')
        plt.axhline(y=50, color='r', linestyle='--', label='50% Ref')
        plt.tight_layout()
        plt.savefig(plot_barras_name)
        print(f"Gráfico guardado: {plot_barras_name}")
        plt.close()
        
    except Exception as e:
        print(f"No se pudo generar el gráfico de barras: {e}")

    # plt.show() # Descomentar si tienes entorno visual

if __name__ == "__main__":
    run_evaluation()