# mus_gym/bots_mus.py
import random
# CORRECCIÓN: Importa desde el paquete local
from .mus_enums import MusAction 


class RandomAgent:
    """
    Este agente elige una acción aleatoria de entre las acciones legales.
    Es el oponente más "tonto" posible.
    """
    def __init__(self, player_id=1):
        self.player_id = player_id
        
    def choose_action(self, observation):
        legal_actions = observation["legal_actions_enums"]
        
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

        if len(legal_actions) == 1:
            return legal_actions[0]

        my_hand_info = observation["my_hand_info"]
        current_lance = observation["game_state"]["current_lance"]
        lance_state = observation["game_state"]["lance_state"]
        
        # --- Lógica de Decisión ---
        
        if lance_state == "START":
            mi_jugada_es_buena = False
            if current_lance == 'GRANDE':
                # Juega "de manual": solo apuesta con Reyes
                mi_jugada_es_buena = (my_hand_info["valores"][0] == 10)
            elif current_lance == 'CHICA':
                # Solo apuesta con Ases
                mi_jugada_es_buena = (my_hand_info["valores"][3] == 1)
            elif current_lance == 'PARES':
                # Apuesta si tiene algo (no es 'Ninguno')
                mi_jugada_es_buena = (my_hand_info["pares"]["tipo"] != 'Ninguno')
            elif current_lance == 'JUEGO':
                 # Apuesta si tiene juego (no es 'Punto')
                mi_jugada_es_buena = (my_hand_info["juego"] != 31) and (my_hand_info["juego"] > 0)
            
            if mi_jugada_es_buena and MusAction.ENVIDO in legal_actions:
                return MusAction.ENVIDO
            else:
                return MusAction.PASO

        elif lance_state == "BET_MADE":
            # Lógica para responder a una apuesta
            mi_jugada_es_buena = False
            if current_lance == 'GRANDE':
                mi_jugada_es_buena = (my_hand_info["valores"][0] == 10)
            elif current_lance == 'CHICA':
                mi_jugada_es_buena = (my_hand_info["valores"][3] == 1)
            elif current_lance == 'PARES':
                mi_jugada_es_buena = (my_hand_info["pares"]["tipo"] != 'Ninguno')
            elif current_lance == 'JUEGO':
                mi_jugada_es_buena = (my_hand_info["juego"] > 0)

            if mi_jugada_es_buena and MusAction.QUIERO in legal_actions:
                return MusAction.QUIERO
            else:
                return MusAction.NO_QUIERO
        
        # Fallback por si algo falla
        return random.choice(legal_actions)