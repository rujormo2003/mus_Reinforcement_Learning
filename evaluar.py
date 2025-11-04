import os
import random
from stable_baselines3 import PPO

# Importamos nuestro entorno y acciones
from entorno_mus import MusEnvironment, MusAction

# --- 1. DEFINICIÓN DE LOS BOTS OPONENTES ---

class RandomAgent:
    """
    Este agente elige una acción aleatoria de entre las acciones legales.
    Es el oponente más "tonto" posible.
    """
    def __init__(self, player_id=1):
        self.player_id = player_id
        
    def choose_action(self, observation):
        legal_actions = observation["legal_actions_enums"] # Usamos la lista de Enums
        
        if not legal_actions:
            return None 
            
        return random.choice(legal_actions)

class BasicoAgent:
    """
    Este agente juega "de manual" (heurístico). No farolea.
    - Apuesta si tiene buena jugada (Reyes/Ases, Pares, Juego).
    - Ve la apuesta si tiene buena jugada.
    - Se retira si no tiene buena jugada.
    """
    def __init__(self, player_id=1):
        self.player_id = player_id
        
    def choose_action(self, observation):
        legal_actions = observation["legal_actions_enums"]
        if not legal_actions:
            return None

        # Si solo hay una acción legal (ej. TENGO/NO_TENGO), tómala
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Extraer información de la observación
        my_hand_info = observation["my_hand_info"]
        current_lance = observation["game_state"]["current_lance"]
        lance_state = observation["game_state"]["lance_state"]
        
        # --- Lógica de Decisión ---
        
        if lance_state == "START":
            # --- Es nuestro turno de ABRIR la apuesta ---
            mi_jugada_es_buena = False
            if current_lance == 'GRANDE':
                # ¿Mi carta más alta es un Rey o 3?
                mi_jugada_es_buena = (my_hand_info["valores"][0] == 10)
            elif current_lance == 'CHICA':
                # ¿Mi carta más baja es un As o 2?
                mi_jugada_es_buena = (my_hand_info["valores"][3] == 1)
            elif current_lance in ['PARES', 'JUEGO']:
                # Ya hemos anunciado "TENGO", así que siempre es "buena"
                mi_jugada_es_buena = True 
            
            # Decisión: Si es buena, envido. Si no, paso.
            if mi_jugada_es_buena and MusAction.ENVIDO in legal_actions:
                return MusAction.ENVIDO
            else:
                return MusAction.PASO

        elif lance_state == "BET_MADE":
            # --- Es nuestro turno de RESPONDER a una apuesta ---
            mi_jugada_es_buena = False
            if current_lance == 'GRANDE':
                mi_jugada_es_buena = (my_hand_info["valores"][0] == 10)
            elif current_lance == 'CHICA':
                mi_jugada_es_buena = (my_hand_info["valores"][3] == 1)
            elif current_lance == 'PARES':
                mi_jugada_es_buena = (my_hand_info["pares"]["tipo"] != 'Ninguno')
            elif current_lance == 'JUEGO':
                mi_jugada_es_buena = my_hand_info["tiene_juego"]

            # Decisión: Si es buena, veo. Si no, me retiro.
            if mi_jugada_es_buena and MusAction.QUIERO in legal_actions:
                return MusAction.QUIERO
            else:
                return MusAction.NO_QUIERO
                
        # Fallback por si algo se escapa (aunque no debería)
        return random.choice(legal_actions)

# --- 2. LA FUNCIÓN DE EVALUACIÓN ---

def evaluar_agente(model, opponent_agent, num_partidas=1000):
    """
    Ejecuta N partidas de la IA (J0) contra un Oponente (J1).
    """
    
    # Importante: necesitamos "engañar" al BasicoAgent
    # dándole la observación que espera.
    # Modificamos el entorno para que la genere.
    
    # Creamos un wrapper (envoltorio) temporal sobre _get_observation
    class EnvWrapper(MusEnvironment):
        def _get_observation(self, player_idx):
            # El modelo IA quiere el vector plano (np.ndarray)
            obs_ia = super()._get_observation(player_idx)
            
            # El bot (humano o heurístico) quiere el dict
            obs_bot = {
                "my_hand_info": {
                    "valores": self.manos[player_idx].get_valores_comparacion(),
                    "pares": self.manos[player_idx].get_pares_info(),
                    "juego": self.manos[player_idx].get_juego_valor(),
                    "tiene_juego": self.manos[player_idx].tiene_juego()
                },
                "game_state": {
                    "scores": self.scores,
                    "is_mano": player_idx == self.mano_player_idx,
                    "current_lance": self.current_lance,
                    "lance_state": self.lance_state,
                },
                "legal_actions_enums": self._get_legal_actions() # Damos los enums
            }
            return obs_ia, obs_bot # Devuelve ambos

    env = EnvWrapper(target_score=30)
    
    ia_wins = 0
    
    print(f"Iniciando evaluación de {num_partidas} partidas...")
    
    for partida in range(num_partidas):
        (obs_ia, obs_bot) = env.reset(seed=partida)[0] # [0] para coger la obs
        done = False
        
        while not done:
            current_player_id = env.current_player_idx
            action_value = None # El int (0-6) que pasaremos a step()
            
            if current_player_id == 0:
                # --- Turno de la IA (J0) ---
                # `deterministic=True` -> La IA usa su *mejor* acción, no explora
                action_tuple = model.predict(obs_ia, deterministic=True)
                action_value = action_tuple[0].item()
            
            else:
                # --- Turno del Bot Oponente (J1) ---
                action_enum = opponent_agent.choose_action(obs_bot)
                if action_enum is None:
                    break # Partida terminada por el otro jugador
                action_value = action_enum.value
            
            # El entorno procesa la acción y devuelve las *nuevas* observaciones
            (obs_ia, obs_bot), reward, terminated, truncated, info = env.step(action_value)
            done = terminated or truncated

        # --- Partida Terminada ---
        if env.scores[0] > env.scores[1]:
            ia_wins += 1
            
        if (partida + 1) % 100 == 0:
            print(f"Partida {partida + 1}/{num_partidas} completada...")

    # --- Evaluación Terminada ---
    tasa_victoria = (ia_wins / num_partidas) * 100
    print("\n--- ¡Evaluación Completa! ---")
    print(f"Partidas jugadas: {num_partidas}")
    print(f"Victorias de la IA (J0): {ia_wins}")
    print(f"Victorias del Bot (J1): {num_partidas - ia_wins}")
    print(f"TASA DE VICTORIA (IA): {tasa_victoria:.2f}%")


# --- 3. EL SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    
    MODEL_PATH = "./models/mus_ppo/mus_agent_1M" # Ajusta si lo guardaste con otro nombre
    
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}.zip")
        print("Asegúrate de haber entrenado el modelo con 'entrenar.py' primero.")
    else:
        # --- Cargar el modelo entrenado ---
        print(f"Cargando modelo desde {MODEL_PATH}...")
        model = PPO.load(MODEL_PATH)
        print("Modelo cargado.")
        
        # --- Selección de Oponente ---
        print("\n¿Contra qué oponente quieres evaluar la IA?")
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

        # --- Iniciar Evaluación ---
        if oponente:
            evaluar_agente(model, oponente, num_partidas=1000)