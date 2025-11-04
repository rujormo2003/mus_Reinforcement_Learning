import os
import random
from stable_baselines3 import PPO

# Importamos nuestro entorno y acciones
from entorno_mus import MusEnvironment, MusAction

# --- 1. DEFINICIÓN DE LOS BOTS OPONENTES (Copiados de evaluar.py) ---

class RandomAgent:
    def __init__(self, player_id=1):
        self.player_id = player_id
    def choose_action(self, observation):
        legal_actions = observation["legal_actions_enums"]
        if not legal_actions: return None 
        return random.choice(legal_actions)

class BasicoAgent:
    def __init__(self, player_id=1):
        self.player_id = player_id
    def choose_action(self, observation):
        legal_actions = observation["legal_actions_enums"]
        if not legal_actions: return None
        if len(legal_actions) == 1: return legal_actions[0]

        my_hand_info = observation["my_hand_info"]
        current_lance = observation["game_state"]["current_lance"]
        lance_state = observation["game_state"]["lance_state"]
        
        if lance_state == "START":
            mi_jugada_es_buena = False
            if current_lance == 'GRANDE': mi_jugada_es_buena = (my_hand_info["valores"][0] == 10)
            elif current_lance == 'CHICA': mi_jugada_es_buena = (my_hand_info["valores"][3] == 1)
            elif current_lance in ['PARES', 'JUEGO']: mi_jugada_es_buena = True 
            
            if mi_jugada_es_buena and MusAction.ENVIDO in legal_actions: return MusAction.ENVIDO
            else: return MusAction.PASO

        elif lance_state == "BET_MADE":
            mi_jugada_es_buena = False
            if current_lance == 'GRANDE': mi_jugada_es_buena = (my_hand_info["valores"][0] == 10)
            elif current_lance == 'CHICA': mi_jugada_es_buena = (my_hand_info["valores"][3] == 1)
            elif current_lance == 'PARES': mi_jugada_es_buena = (my_hand_info["pares"]["tipo"] != 'Ninguno')
            elif current_lance == 'JUEGO': mi_jugada_es_buena = my_hand_info["tiene_juego"]

            if mi_jugada_es_buena and MusAction.QUIERO in legal_actions: return MusAction.QUIERO
            else: return MusAction.NO_QUIERO
                
        return random.choice(legal_actions)

# --- 2. FUNCIÓN DE PARTIDA VERBOSE ---

def jugar_partida_verbose(model, opponent_agent):
    
    env = MusEnvironment(target_score=30)
    observation, info = env.reset(seed=random.randint(0, 10000)) # Partida aleatoria
    done = False
    
    print("=" * 40)
    print(f"Iniciando partida: IA (J0) vs {opponent_agent.__class__.__name__} (J1)")
    print("=" * 40)

    while not done:
        # 1. Determinar jugador y mostrar estado
        current_player_id = env.current_player_idx
        
        print("\n" + "-" * 30)
        print(f"PUNTUACIÓN: [IA: {env.scores[0]} | Bot: {env.scores[1]}]")
        print(f"Turno de: J{current_player_id} ({'IA' if current_player_id == 0 else 'Bot'})")
        print(f"Posición: J{env.mano_player_idx} es 'mano'")
        print(f"Lance: {env.current_lance} | Estado: {env.lance_state}")
        
        # 2. ¡MOSTRAR AMBAS MANOS! (La clave del debug)
        print(f"  Mano J0 (IA): {env.manos[0]}")
        print(f"  Mano J1 (Bot): {env.manos[1]}")

        # 3. Obtener acciones legales
        legal_actions = env._get_legal_actions()
        print(f"  Acciones Legales: {[a.name for a in legal_actions]}")

        action_value = None
        action_enum = None
        
        if current_player_id == 0:
            # --- Turno de la IA (J0) ---
            obs_ia_vector = env._get_observation(current_player_id)
            action_tuple = model.predict(obs_ia_vector, deterministic=True)
            action_value = action_tuple[0].item()
            action_enum = MusAction(action_value)
            
        else:
            # --- Turno del Bot Oponente (J1) ---
            # Creamos la observación "humana" que el bot espera
            obs_bot = {
                "my_hand_info": {
                    "valores": env.manos[1].get_valores_comparacion(),
                    "pares": env.manos[1].get_pares_info(),
                    "juego": env.manos[1].get_juego_valor(),
                    "tiene_juego": env.manos[1].tiene_juego()
                },
                "game_state": {
                    "current_lance": env.current_lance,
                    "lance_state": env.lance_state,
                },
                "legal_actions_enums": legal_actions
            }
            action_enum = opponent_agent.choose_action(obs_bot)
            if action_enum is None:
                break 
            action_value = action_enum.value
        
        # 4. Mostrar la acción elegida
        print(f"\n  ACCIÓN ELEGIDA: >> {action_enum.name} <<")
        
        # 5. Ejecutar el paso
        observation, reward, terminated, truncated, info = env.step(action_value)
        done = terminated or truncated

        if reward != 0:
            print(f"  *** J{current_player_id} recibe recompensa: {reward} ***")
            
        if info.get("error"):
            print(f"  !!! ERROR DE ENTORNO: {info['error']} !!!")

        # ¡PAUSA!
        input("\n...Pulsa Enter para el siguiente turno...")

    # --- Fin de la partida ---
    print("\n" + "=" * 40)
    print("¡¡¡PARTIDA TERMINADA!!!")
    print(f"PUNTUACIÓN FINAL: [IA: {env.scores[0]} | Bot: {env.scores[1]}]")
    if env.scores[0] > env.scores[1]:
        print("GANADOR: IA (J0)")
    else:
        print("GANADOR: Bot (J1)")
    print("=" * 40)


# --- 3. EL SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    
    MODEL_PATH = "./models/mus_ppo/mus_agent_1M" # Ajusta si lo guardaste con otro nombre
    
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}.zip")
    else:
        print(f"Cargando modelo desde {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH)
        print("Modelo cargado.")
        
        print("\n¿Contra qué oponente quieres jugar la partida de depuración?")
        print("  1: Bot Aleatorio (muy fácil)")
        print("  2: Bot Básico (juega de manual, sin faroles)")
        
        oponente = None
        choice = input("Elige una opción (1 o 2): ")
        
        if choice == '1':
            print("Seleccionado: Bot Aleatorio")
            oponente = RandomAgent(player_id=1)
        elif choice == '2':
            print("Seleccionado: Bot Básico (Heurístico)")
            oponente = BasicoAgent(player_id=1)
        else:
            print("Opción no válida.")

        if oponente:
            jugar_partida_verbose(model, oponente)