import random
from .mus_enums import MusAction 

class HeuristicBot:
    """
    Agente heurístico robusto.
    MODIFICADO: Más agresivo en Grande y Chica.
    """
    def __init__(self, player_id=1, prob_farol=0.15, prob_error=0.05, prob_hero_call=0.10):
        self.player_id = player_id
        self.prob_farol = prob_farol      
        self.prob_error = prob_error      
        self.prob_hero_call = prob_hero_call 
        
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
            
            # --- DEFINICIÓN DE UMBRALES ---
            # Por defecto (Pares y Juego) exigimos fuerza 0.6
            umbral_envite = 0.60
            
            # Para Grande y Chica bajamos la exigencia a 0.45
            # Esto hará que envide con manos "normalitas"
            if lance in ['GRANDE', 'CHICA']:
                umbral_envite = 0.45

            # ------------------------------

            # Si es muy fuerte (Top 10%), tirar Ordago o Envido
            if fuerza > 0.90:
                if MusAction.ORDAGO in legal_actions and random.random() < 0.5:
                    return MusAction.ORDAGO
                return MusAction.ENVIDO if MusAction.ENVIDO in legal_actions else MusAction.PASO
            
            # Mano Decente (Usamos el umbral dinámico)
            if fuerza > umbral_envite:
                return MusAction.ENVIDO if MusAction.ENVIDO in legal_actions else MusAction.PASO
            
            # Mano Mala (Posible Farol)
            # Nota: Si el umbral_envite es bajo (0.45), el espacio para 'PASO' se reduce,
            # haciendo el bot mucho más activo.
            if fuerza < 0.4:
                if random.random() < self.prob_farol and MusAction.ENVIDO in legal_actions:
                    return MusAction.ENVIDO 
                return MusAction.PASO
            
            # Caso intermedio (Mano ni fu ni fa -> PASO)
            return MusAction.PASO

        elif state == "BET_MADE":
            
            # Lógica para RESPONDER apuestas (NO_QUIERO / QUIERO / ORDAGO)
            
            # Si nos tiran Órdago
            if observation["game_state"]["bet"] >= 30: 
                
                # Nivel 1: Mano Invencible
                if fuerza >= 0.95:
                    return MusAction.QUIERO
                
                # Nivel 2: Hero Call
                if 0.5 <= fuerza < 0.95:
                    if random.random() < self.prob_hero_call:
                        return MusAction.QUIERO 
                
                return MusAction.NO_QUIERO

            # Si es un Envido normal
            if fuerza > 0.75:
                if MusAction.ORDAGO in legal_actions and random.random() < 0.1:
                    return MusAction.ORDAGO
                return MusAction.QUIERO
            
            # Bajamos un poco también la exigencia para QUERER envites en Grande/Chica
            umbral_quiero = 0.5
            if lance in ['GRANDE', 'CHICA']:
                umbral_quiero = 0.45

            if fuerza > umbral_quiero:
                return MusAction.QUIERO 
            
            return MusAction.NO_QUIERO

        return MusAction.PASO

    def _calcular_fuerza(self, lance, hand) -> float:
        """ Devuelve un valor 0.0 a 1.0 estimando la fuerza de la mano """
        vals = hand["valores"]
        
        if lance == 'GRANDE':
            suma = sum(vals)
            # Ajuste matemático: Max 40, Min 4. Rango 36.
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