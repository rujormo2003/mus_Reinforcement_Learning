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
# mus_gym/mus_bots.py (Actualizado)

import random
from .mus_enums import MusAction 

class HeuristicBot:
    """
    Agente heurístico robusto con mayor riesgo de farol y capacidad de "Hero Call" 
    para forzar al agente de RL a aprender cuándo retirarse o seguir apostando.
    """
    def __init__(self, player_id=1, prob_farol=0.15, prob_error=0.05, prob_hero_call=0.10):
        self.player_id = player_id
        self.prob_farol = prob_farol      # 15% de veces miente con mano mala (Envido)
        self.prob_error = prob_error      # 5% de jugadas totalmente aleatorias (Ruido)
        self.prob_hero_call = prob_hero_call # 10% de veces llama Órdagos con mano media
        
    def choose_action(self, observation):
        legal_actions = observation["legal_actions_enums"]
        if not legal_actions: return None
        if len(legal_actions) == 1: return legal_actions[0]

        # 1. Ruido Aleatorio (Epsilon-Greedy)
        if random.random() < self.prob_error:
            return random.choice(legal_actions)

        hand_info = observation["my_hand_info"]
        lance = observation["game_state"]["current_lance"]
        state = observation["game_state"]["lance_state"]
        fuerza = self._calcular_fuerza(lance, hand_info)
        
        # --- 3. Toma de Decisión ---
        
        if state == "START":
            # Si es muy fuerte, tirar Ordago o Envido
            if fuerza > 0.90:
                if MusAction.ORDAGO in legal_actions and random.random() < 0.5:
                    return MusAction.ORDAGO
                return MusAction.ENVIDO if MusAction.ENVIDO in legal_actions else MusAction.PASO
            
            # Mano Decente
            if fuerza > 0.6:
                return MusAction.ENVIDO if MusAction.ENVIDO in legal_actions else MusAction.PASO
            
            # Mano Mala (Posible Farol)
            if fuerza < 0.4:
                if random.random() < self.prob_farol and MusAction.ENVIDO in legal_actions:
                    return MusAction.ENVIDO 
                return MusAction.PASO
            
            # Caso intermedio
            return MusAction.PASO

        elif state == "BET_MADE":
            
            # Lógica para RESPONDER apuestas (NO_QUIERO / QUIERO / ORDAGO)
            
            # Si nos tiran Órdago
            if observation["game_state"]["bet"] >= 30: 
                
                # Nivel 1: Mano Invencible (Ej: 31, 4 Reyes, Duples máximos)
                if fuerza >= 0.95:
                    return MusAction.QUIERO
                
                # Nivel 2: Mano Media/Fuerte (Fuerza 0.5 a 0.85) - HERO CALL
                if 0.5 <= fuerza < 0.95:
                    if random.random() < self.prob_hero_call:
                        return MusAction.QUIERO # ¡Castigamos el posible farol!
                
                # Nivel 3: Mano Débil o Nivel 2 fallido
                return MusAction.NO_QUIERO

            # Si es un Envido normal
            if fuerza > 0.75:
                # 10% de veces contrataca con Órdago si es legal
                if MusAction.ORDAGO in legal_actions and random.random() < 0.1:
                    return MusAction.ORDAGO
                return MusAction.QUIERO
            
            # Mano Media/Baja
            if fuerza > 0.5:
                return MusAction.QUIERO 
            
            return MusAction.NO_QUIERO

        return MusAction.PASO

    def _calcular_fuerza(self, lance, hand) -> float:
        """ Devuelve un valor 0.0 a 1.0 estimando la fuerza de la mano """
        vals = hand["valores"]
        
        if lance == 'GRANDE':
            suma = sum(vals)
            return (suma - 4) / 36.0

        elif lance == 'CHICA':
            suma = sum(vals)
            return 1.0 - ((suma - 4) / 36.0)

        elif lance == 'PARES':
            info = hand["pares"]
            if info["tipo"] == 'Ninguno': return 0.0
            
            base = 0.0
            if info["tipo"] == 'Pares': base = 0.4
            elif info["tipo"] == 'Medias': base = 0.7
            elif info["tipo"] == 'Duples': base = 0.9
            
            plus = info["valor_principal"] / 100.0
            return min(1.0, base + plus)

        elif lance == 'JUEGO':
            pts = hand["juego"]
            if pts == 0: return 0.0
            
            ranking = [33, 34, 35, 36, 37, 40, 32, 31]
            if pts in ranking:
                idx = ranking.index(pts)
                return (idx + 1) / len(ranking)
            return 0.0

        return 0.0