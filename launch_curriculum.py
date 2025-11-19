# launch_curriculum.py
import subprocess
import os
import sys
from datetime import datetime

# --- 1. CONFIGURACIÓN ---
# RUTA AL MODELO OPONENTE ENTRENADO
# 🛑 SUSTITUYE ESTA RUTA por el path de tu modelo final (ej. models/ppo_lstm_mus_v3_vec_...)
OPPONENT_MODEL_PATH = "models/ppo_lstm_mus_v3_vec_.zip"  
SCRIPT_TO_RUN = "train_curriculum.py"

# --- 2. DEFINICIÓN DE LAS 4 CORRIDAS ---
BASE_NAME = f"Curriculum_{datetime.now().strftime('%Y%m%d')}"

RUN_CONFIGS = [
    # 1. Conservador (Explota las agresiones del oponente)
    {"id": 1, "ent_coef": 0.01, "lr": 0.0001, "gamma": 0.99},
    
    # 2. Balanceado (Estrategia media, más probable que funcione)
    {"id": 2, "ent_coef": 0.04, "lr": 0.0002, "gamma": 0.995},
    
    # 3. Agresivo (Alta exploración, valora ganar a largo plazo)
    {"id": 3, "ent_coef": 0.08, "lr": 0.0003, "gamma": 0.999},
    
    # 4. Velocidad/Alto Riesgo (LR más alta, muy agresivo)
    {"id": 4, "ent_coef": 0.06, "lr": 0.0005, "gamma": 0.995},
]

def launch_run(config):
    """ Construye y lanza el comando para una sola corrida. """
    run_name = f"{BASE_NAME}_R{config['id']}_{config['ent_coef']}"
    
    command = [
        sys.executable,  
        SCRIPT_TO_RUN,
        f"--run_name={run_name}",
        f"--opponent_path={OPPONENT_MODEL_PATH}",
        f"--ent_coef={config['ent_coef']}",
        f"--learning_rate={config['lr']}",
        f"--gamma={config['gamma']}"
    ]
    
    print(f"Lanzando entrenamiento: {run_name}...")
    
    # Lanza el proceso sin esperar a que termine
    return subprocess.Popen(command, cwd=os.getcwd())

if __name__ == "__main__":
    if not os.path.exists(OPPONENT_MODEL_PATH):
        print(f"ERROR: No se encontró el modelo oponente en {OPPONENT_MODEL_PATH}. Por favor, corrige la ruta.")
        sys.exit(1)

    processes = []
    
    print(f"Iniciando {len(RUN_CONFIGS)} entrenamientos SIMULTÁNEOS...")
    
    for config in RUN_CONFIGS:
        p = launch_run(config)
        processes.append(p)

    print("\n--- Todos los procesos han sido lanzados. Monitorea con TensorBoard. ---")
    print("\nEsperando finalización de los procesos...")

    for i, p in enumerate(processes):
        p.wait()
        config_name = RUN_CONFIGS[i]['name']
        
        if p.returncode == 0:
            print(f"✅ Proceso {config_name} finalizado con éxito.")
        else:
            print(f"❌ ERROR: Proceso {config_name} falló (Código: {p.returncode}).")
            
    print("\nTodos los entrenamientos simultáneos han concluido.")