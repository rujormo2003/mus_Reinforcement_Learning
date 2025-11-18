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

class HeuristicBot:
    """
    Agente heurístico robusto para entrenamiento.
    - Evalúa la fuerza de la mano numéricamente.
    - Tiene probabilidad de farolear (bluff).
    - Tira órdagos con manos 'monstruo' o muy raramente de farol.
    - Introduce ruido estocástico para no ser determinista.
    """
    def __init__(self, player_id=1, prob_farol=0.10, prob_error=0.05, agresividad=0.2):
        self.player_id = player_id
        self.prob_farol = prob_farol   # 10% de las veces miente con mano mala
        self.prob_error = prob_error   # 5% de jugadas totalmente aleatorias (ruido)
        self.agresividad = agresividad # Factor para decidir si envidar/querer en dudosas

    def choose_action(self, observation):
        legal_actions = observation["legal_actions_enums"]
        if not legal_actions:
            return None
        if len(legal_actions) == 1:
            return legal_actions[0]

        # --- 1. Ruido Aleatorio (Epsilon-Greedy) ---
        # Ayuda a que tu agente de RL explore estados raros
        if random.random() < self.prob_error:
            return random.choice(legal_actions)

        # --- 2. Análisis de la situación ---
        hand_info = observation["my_hand_info"]
        lance = observation["game_state"]["current_lance"]
        state = observation["game_state"]["lance_state"]
        
        # Calculamos la fuerza de la mano entre 0.0 (Pésima) y 1.0 (Invencible)
        fuerza = self._calcular_fuerza(lance, hand_info)
        
        # --- 3. Toma de Decisión ---
        
        if state == "START":
            # Lógica para ABRIR apuestas (PASO / ENVIDO / ORDAGO)
            
            # Caso: Mano MUY fuerte (Órdago o Envido seguro)
            if fuerza > 0.90:
                if MusAction.ORDAGO in legal_actions and random.random() < 0.4:
                    return MusAction.ORDAGO
                return MusAction.ENVIDO if MusAction.ENVIDO in legal_actions else MusAction.PASO
            
            # Caso: Mano Decente (Envido por valor)
            if fuerza > 0.6:
                return MusAction.ENVIDO if MusAction.ENVIDO in legal_actions else MusAction.PASO
            
            # Caso: Mano Mala (Posible Farol)
            if fuerza < 0.4:
                if random.random() < self.prob_farol and MusAction.ENVIDO in legal_actions:
                    return MusAction.ENVIDO # ¡Farol!
                return MusAction.PASO
            
            # Caso intermedio: Juega conservador pero a veces ataca
            if random.random() < self.agresividad and MusAction.ENVIDO in legal_actions:
                return MusAction.ENVIDO
            return MusAction.PASO

        elif state == "BET_MADE":
            # Lógica para RESPONDER apuestas (NO_QUIERO / QUIERO / ORDAGO)
            
            # Si nos tiran Órdago
            if observation["game_state"]["bet"] >= 30: # Asumiendo bet alto es órdago
                # Solo queremos con manos premium (o un farol suicida muy raro)
                limit = 0.85
                if lance == 'JUEGO': limit = 0.95 # En juego hay que ser estricto (31)
                
                if fuerza >= limit:
                    return MusAction.QUIERO
                return MusAction.NO_QUIERO

            # Si es un Envido normal
            # Si la mano es buena, queremos y a veces re-envidamos (si se pudiera) u Ordago
            if fuerza > 0.75:
                # 10% de veces contrataca con Órdago si es legal
                if MusAction.ORDAGO in legal_actions and random.random() < 0.1:
                    return MusAction.ORDAGO
                return MusAction.QUIERO
            
            # Si la mano es media/baja
            if fuerza > 0.5:
                return MusAction.QUIERO # Pagar para ver
            
            # Farol de contra-ataque (Bluff catch o re-bluff)
            if random.random() < (self.prob_farol / 2) and MusAction.ORDAGO in legal_actions:
                 return MusAction.ORDAGO

            return MusAction.NO_QUIERO

        return MusAction.PASO

    def _calcular_fuerza(self, lance, hand) -> float:
        """ Devuelve un valor 0.0 a 1.0 estimando la fuerza de la mano """
        vals = hand["valores"] # Lista de 4 enteros
        
        if lance == 'GRANDE':
            # Sumar valor de cartas. Max posible 4 Reyes (10*4=40). Min Ases (4).
            # Normalizamos: (Suma - 4) / 36
            suma = sum(vals)
            return (suma - 4) / 36.0

        elif lance == 'CHICA':
            # Inverso a grande. Queremos valores bajos.
            # Max fuerza = suma 4. Min fuerza = suma 40.
            suma = sum(vals)
            return 1.0 - ((suma - 4) / 36.0)

        elif lance == 'PARES':
            # Jerarquía: Duples > Medias > Pares > Nada
            info = hand["pares"]
            if info["tipo"] == 'Ninguno': return 0.0
            
            # Base score
            base = 0.0
            if info["tipo"] == 'Pares': base = 0.4
            elif info["tipo"] == 'Medias': base = 0.7
            elif info["tipo"] == 'Duples': base = 0.9
            
            # Ajuste fino por el valor de las cartas (Duples de Reyes > Duples de Ases)
            plus = info["valor_principal"] / 100.0 # Pequeño extra
            return min(1.0, base + plus)

        elif lance == 'JUEGO':
            # Jerarquía estricta: 31 > 32 > 40 > 37 > 36 > 35 > 34 > 33
            pts = hand["juego"]
            if pts == 0: return 0.0 # Tiene punto, no juego
            
            ranking = [33, 34, 35, 36, 37, 40, 32, 31]
            if pts in ranking:
                idx = ranking.index(pts)
                return (idx + 1) / len(ranking) # 31 -> 1.0, 33 -> 0.125
            return 0.0

        return 0.0