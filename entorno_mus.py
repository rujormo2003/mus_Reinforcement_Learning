# --- IMPORTS ---
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from enum import Enum
from collections import Counter

# --- 1. Definición de Acciones (¡MODIFICADO!) ---

class MusAction(Enum):
    PASO = 0
    ENVIDO = 1
    ORDAGO = 2
    QUIERO = 3
    NO_QUIERO = 4
    # Se eliminaron TENGO y NO_TENGO
    # El agente ahora solo tiene 5 acciones.

# --- 2. Clases de Cartas, Baraja y Mano (Sin cambios) ---

PALOS = ['Oros', 'Copas', 'Espadas', 'Bastos']
RANGOS_REALES = ['As', '2', '3', '4', '5', '6', '7', 'Sota', 'Caballo', 'Rey']
VALORES_JUEGO = {'As': 1, '2': 1, '3': 10, '4': 4, '5': 5, '6': 6, '7': 7, 'Sota': 10, 'Caballo': 10, 'Rey': 10}
VALORES_COMPARACION = {'As': 1, '2': 1, '3': 10, '4': 4, '5': 5, '6': 6, '7': 7, 'Sota': 8, 'Caballo': 9, 'Rey': 10}

class Carta:
    def __init__(self, palo, rango):
        self.palo = palo
        self.rango = rango
        self.valor_juego = VALORES_JUEGO[rango]
        self.valor_comparacion = VALORES_COMPARACION[rango]
    def __repr__(self): return f"{self.rango} de {self.palo}"

class Baraja:
    def __init__(self):
        self.cartas = []
        self.crear_baraja()
    def crear_baraja(self):
        self.cartas = [Carta(palo, rango) for palo in PALOS for rango in RANGOS_REALES]
    def mezclar(self): random.shuffle(self.cartas)
    def repartir(self, num_cartas):
        if len(self.cartas) < num_cartas:
            self.crear_baraja(); self.mezclar()
        return [self.cartas.pop() for _ in range(num_cartas)]

class Mano:
    def __init__(self, cartas: list[Carta]):
        self.cartas = cartas
        self.cartas_ordenadas = sorted(self.cartas, key=lambda c: c.valor_comparacion, reverse=True)
        self._valores_comparacion = [c.valor_comparacion for c in self.cartas_ordenadas]
        self._valor_juego_total = sum(c.valor_juego for c in self.cartas)
        self.tiene_juego_bool = self._valor_juego_total >= 31
        
        conteo = Counter(self._valores_comparacion)
        self.tiene_pares_bool = False
        self._pares_info = {'tipo': 'Ninguno', 'valor_principal': 0, 'valor_secundario': 0}
        valores_conteo = conteo.values()

        if 4 in valores_conteo:
            self.tiene_pares_bool = True
            valor = list(conteo.keys())[0]
            self._pares_info = {'tipo': 'Duples', 'valor_principal': valor, 'valor_secundario': valor}
        elif 3 in valores_conteo:
            self.tiene_pares_bool = True
            valor = [v for v, c in conteo.items() if c == 3][0]
            self._pares_info = {'tipo': 'Medias', 'valor_principal': valor}
        elif 2 in valores_conteo:
            pares = [v for v, c in conteo.items() if c == 2]
            if len(pares) == 2:
                self.tiene_pares_bool = True
                self._pares_info = {'tipo': 'Duples', 'valor_principal': max(pares), 'valor_secundario': min(pares)}
            else:
                self.tiene_pares_bool = True
                self._pares_info = {'tipo': 'Pares', 'valor_principal': pares[0]}
                
    def __repr__(self): return f"Mano({[str(c) for c in self.cartas]})"
    def get_valores_comparacion(self): return self._valores_comparacion
    def tiene_juego(self): return self.tiene_juego_bool
    def get_juego_valor(self): return self._valor_juego_total
    def tiene_pares(self): return self.tiene_pares_bool
    def get_pares_info(self): return self._pares_info


# --- 3. El Entorno "Gym-ificado" ---

class MusEnvironment(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4} 

    def __init__(self, target_score=30):
        super().__init__() 
        
        self.target_score = target_score
        self.baraja = Baraja()
        
        # --- Estado Global ---
        self.scores = {0: 0, 1: 0}
        self.mano_player_idx = 0
        self.game_over = False
        
        # --- Estado de Mano ---
        self.manos = {0: None, 1: None}
        self.historial_acciones = [] 
        self.current_player_idx = 0
        self.current_lance = None
        self.lance_bet = 0
        self.player_who_bet = None
        self.lance_state = "START"
        # self.anuncios fue eliminado

        # --- ¡MODIFICADO! Definición de Espacios ---
        
        # 1. Espacio de Acciones: 5 acciones discretas (0-4)
        self.action_space = spaces.Discrete(len(MusAction)) # Ahora es 5
        
        # 2. Espacio de Observación: 28 números
        # Se quitó 1 estado de 'estados_lance' ('ANNOUNCE')
        self.observation_space_size = 28 
        max_obs_value = max(40, self.target_score) 
        
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=max_obs_value, 
            shape=(self.observation_space_size,), # ¡Cambiado a 28!
            dtype=np.float32
        )
        
    def _get_winner(self, idx_a, idx_b, is_winner_a):
        # (Sin cambios)
        if is_winner_a is None: return self.mano_player_idx
        return idx_a if is_winner_a else idx_b
        
    # --- Funciones de Comparación (Sin cambios) ---
    def _compare_grande(self):
        vals0=self.manos[0].get_valores_comparacion();vals1=self.manos[1].get_valores_comparacion()
        w0=None;
        for i in range(4):
            if vals0[i]>vals1[i]:w0=True;break
            if vals0[i]<vals1[i]:w0=False;break
        return self._get_winner(0,1,w0)
    def _compare_chica(self):
        vals0=self.manos[0].get_valores_comparacion();vals1=self.manos[1].get_valores_comparacion()
        w0=None
        for i in range(3,-1,-1):
            if vals0[i]<vals1[i]:w0=True;break
            if vals0[i]>vals1[i]:w0=False;break
        return self._get_winner(0,1,w0)
    def _compare_pares(self):
        info0,info1=self.manos[0].get_pares_info(),self.manos[1].get_pares_info()
        rank={'Ninguno':0,'Pares':1,'Medias':2,'Duples':3}
        r0,r1=rank[info0['tipo']],rank[info1['tipo']];w0=None
        if r0>r1:w0=True
        elif r0<r1:w0=False
        else:
            if info0['valor_principal']>info1['valor_principal']:w0=True
            elif info0['valor_principal']<info1['valor_principal']:w0=False
            elif info0['tipo']=='Duples' and w0 is None:
                if info0.get('valor_secundario',0)>info1.get('valor_secundario',0):w0=True
                elif info0.get('valor_secundario',0)<info1.get('valor_secundario',0):w0=False
        return self._get_winner(0,1,w0)
    def _compare_juego(self):
        j0,j1=self.manos[0].get_juego_valor(),self.manos[1].get_juego_valor()
        tj0,tj1=self.manos[0].tiene_juego(),self.manos[1].tiene_juego();w0=None
        if tj0 and not tj1:w0=True
        elif not tj0 and tj1:w0=False
        elif tj0 and tj1:
            if j0==31:j0=100
            if j1==31:j1=100
            if j0==32:j0=99
            if j1==32:j1=99
            if j0>j1:w0=True
            if j0<j1:w0=False
        elif not tj0 and not tj1:
            if j0>j1:w0=True
            if j0<j1:w0=False
        return self._get_winner(0,1,w0)
    
    # --- ¡LÓGICA DE TRANSICIÓN MODIFICADA! ---
    
    def _set_next_lance(self, lance_name):
        """
        Prepara el estado para un nuevo lance que SÍ requiere
        la intervención del agente (GRANDE, CHICA, o PARES/JUEGO si ambos tienen).
        """
        self.current_lance = lance_name
        self.current_player_idx = self.mano_player_idx
        self.lance_bet = 0
        self.player_who_bet = None
        self.lance_state = "START"
        # self.anuncios fue eliminado

    def _move_to_next_lance(self):
        """
        ¡NUEVA LÓGICA!
        Avanza al siguiente lance, resolviendo automáticamente PARES y JUEGO
        si no hay envite (si no tienen ambos).
        """
        if self.game_over: 
            return

        lances = ['GRANDE', 'CHICA', 'PARES', 'JUEGO']
        
        current_idx = -1
        try:
            current_idx = lances.index(self.current_lance)
        except ValueError:
            current_idx = -1 # Pasa si self.current_lance es None (inicio)
        
        next_lance_idx = current_idx + 1

        # Bucle 'while' para saltar lances auto-resueltos
        while next_lance_idx < len(lances):
            next_lance_name = lances[next_lance_idx]
            
            lance_resolved_and_points_given = False

            if next_lance_name == 'PARES':
                p0_has = self.manos[0].tiene_pares()
                p1_has = self.manos[1].tiene_pares()

                if p0_has and p1_has:
                    # Ambos tienen: Inicia el lance y detiene el bucle
                    self._set_next_lance(next_lance_name)
                    return
                elif p0_has and not p1_has:
                    self.scores[0] += 1
                    lance_resolved_and_points_given = True
                elif not p0_has and p1_has:
                    self.scores[1] += 1
                    lance_resolved_and_points_given = True
                # Si 'else' (ninguno tiene), no se dan puntos, el lance se salta
                
            elif next_lance_name == 'JUEGO':
                p0_has = self.manos[0].tiene_juego()
                p1_has = self.manos[1].tiene_juego()

                if p0_has and p1_has:
                    # Ambos tienen: Inicia el lance y detiene el bucle
                    self._set_next_lance(next_lance_name)
                    return
                elif p0_has and not p1_has:
                    self.scores[0] += 1
                    lance_resolved_and_points_given = True
                elif not p0_has and p1_has:
                    self.scores[1] += 1
                    lance_resolved_and_points_given = True
                # Si 'else' (ninguno tiene), no se dan puntos, el lance se salta

            else: # Es 'GRANDE' o 'CHICA'
                self._set_next_lance(next_lance_name)
                return

            # Si se dieron puntos, comprobar si el juego ha terminado
            if lance_resolved_and_points_given:
                self._check_game_over()
                if self.game_over:
                    return
            
            # Si el lance fue 'PARES' o 'JUEGO' y se resolvió,
            # el bucle 'while' continúa con next_lance_idx + 1
            next_lance_idx += 1
        
        # Si el bucle 'while' termina, significa que se han procesado todos los lances
        self._start_new_hand()

    # --- Fin de la Lógica de Transición ---

    def _resolve_lance_winner(self):
        # (Sin cambios)
        if self.current_lance == 'GRANDE': return self._compare_grande()
        if self.current_lance == 'CHICA': return self._compare_chica()
        if self.current_lance == 'PARES': return self._compare_pares()
        if self.current_lance == 'JUEGO': return self._compare_juego()

    def _check_game_over(self):
        # (Sin cambios)
        if self.scores[0] >= self.target_score or self.scores[1] >= self.target_score:
            self.game_over = True

    def _get_next_player(self): 
        # (Sin cambios)
        return 1 - self.current_player_idx

    # --- ¡ELIMINADO! ---
    # La función _handle_action_announce fue borrada.

    def _handle_action_start(self, action):
        # (Sin cambios)
        player = self.current_player_idx
        if action == MusAction.PASO:
            if self.current_player_idx == self.mano_player_idx: self.current_player_idx = self._get_next_player()
            else: self._move_to_next_lance()
        elif action == MusAction.ENVIDO:
            self.lance_state = "BET_MADE"; self.lance_bet = 2; self.player_who_bet = player
            self.current_player_idx = self._get_next_player()
        elif action == MusAction.ORDAGO:
            self.lance_state = "BET_MADE"; self.lance_bet = self.target_score; self.player_who_bet = player
            self.current_player_idx = self._get_next_player()
            
    def _handle_action_response(self, action):
        # (Sin cambios)
        responder = self.current_player_idx; better = self.player_who_bet; 
        reward = 0.0

        if action == MusAction.QUIERO:
            winner = self._resolve_lance_winner(); points = self.lance_bet
            self.scores[winner] += points
            if winner == responder: reward = points / self.target_score
            else: reward = -points / self.target_score
            
        elif action == MusAction.NO_QUIERO:
            self.scores[better] += 1
            reward = -1.0 / self.target_score 
            
        elif action == MusAction.ORDAGO:
            if self.lance_bet == self.target_score: 
                return self._handle_action_response(MusAction.QUIERO)
            else: 
                self.lance_state = "BET_MADE"; self.lance_bet = self.target_score
                self.player_who_bet = responder; self.current_player_idx = better
                return 0.0 

        self._check_game_over(); self._move_to_next_lance()
        return reward

    def _start_new_hand(self):
        # (Sin cambios)
        self.mano_player_idx = 1 - self.mano_player_idx
        self.baraja.crear_baraja(); self.baraja.mezclar()
        self.manos[0] = Mano(self.baraja.repartir(4))
        self.manos[1] = Mano(self.baraja.repartir(4))
        self.historial_acciones = [] 
        
        # --- ¡CAMBIO! ---
        # Al iniciar la mano, llamamos a move_to_next_lance 
        # que auto-resolverá hasta GRANDE (o PARES/JUEGO si los primeros se saltan)
        self.current_lance = None # Aseguramos que empiece desde el índice -1
        self._move_to_next_lance()

    def _get_legal_actions(self) -> list[MusAction]:
        # --- ¡MODIFICADO! ---
        # Se eliminó el bloque "ANNOUNCE"
        if self.lance_state == "START":
            return [MusAction.PASO, MusAction.ENVIDO, MusAction.ORDAGO]
        if self.lance_state == "BET_MADE":
            if self.lance_bet == self.target_score:
                return [MusAction.QUIERO, MusAction.NO_QUIERO]
            else:
                return [MusAction.QUIERO, MusAction.NO_QUIERO, MusAction.ORDAGO]
        return [] # Estado bloqueado (no debería ocurrir)


    def _get_observation(self, player_idx) -> np.ndarray:
        """
        ¡MODIFICADO! Construye el vector de observación de 28 números.
        """
        obs = np.zeros(self.observation_space_size, dtype=np.float32)
        
        # --- 1. Info de Mano (11 números) [Índices 0-10] ---
        # (Sin cambios)
        mano = self.manos[player_idx]
        obs[0:4] = mano.get_valores_comparacion()
        pares_info = mano.get_pares_info()
        if pares_info['tipo'] == 'Pares':   obs[4] = 1.0
        elif pares_info['tipo'] == 'Medias':  obs[5] = 1.0
        elif pares_info['tipo'] == 'Duples':  obs[6] = 1.0
        obs[7] = pares_info['valor_principal']
        obs[8] = pares_info.get('valor_secundario', 0)
        obs[9] = 1.0 if mano.tiene_juego() else 0.0
        obs[10] = mano.get_juego_valor()
        
        # --- 2. Info de Juego (11 números) [Índices 11-21] ---
        obs[11] = self.scores[player_idx]
        obs[12] = self.scores[1 - player_idx] 
        obs[13] = 1.0 if player_idx == self.mano_player_idx else 0.0
        
        # One-hot para el lance [Índices 14-17] (4 números)
        lances = ['GRANDE', 'CHICA', 'PARES', 'JUEGO']
        if self.current_lance in lances:
            obs[14 + lances.index(self.current_lance)] = 1.0
            
        # One-hot para el estado del lance [Índices 18-19] (2 números)
        estados_lance = ['START', 'BET_MADE'] # 'ANNOUNCE' eliminado
        if self.lance_state in estados_lance:
            obs[18 + estados_lance.index(self.lance_state)] = 1.0

        # Info de apuesta [Índices 20-21] (2 números)
        obs[20] = self.lance_bet
        if self.player_who_bet is not None:
            obs[21] = 1.0 if self.player_who_bet == player_idx else -1.0
            
        # --- 3. Historial (6 números) [Índices 22-27] ---
        hist_len = len(self.historial_acciones)
        if hist_len > 0:
            start = max(0, hist_len - 6)
            end = hist_len
            hist_slice = self.historial_acciones[start:end]
            # Rellenar en el vector de obs (índice 22)
            obs[22 : 22 + len(hist_slice)] = hist_slice
            # Rellenar el resto con -1 (padding)
            if len(hist_slice) < 6:
                obs[22 + len(hist_slice) : 28] = -1.0 # 22 + 6 = 28

        return obs

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno para Gym.
        """
        super().reset(seed=seed) 
        
        self.scores = {0: 0, 1: 0}
        self.mano_player_idx = random.choice([0, 1])
        self.game_over = False
        
        # --- ¡CAMBIO! ---
        # _start_new_hand() ahora maneja la auto-resolución
        self._start_new_hand() 
        
        observation = self._get_observation(self.current_player_idx)
        info = {} 
        
        return observation, info

    def step(self, action_value):
        """
        ¡MODIFICADO!
        Procesa una acción del agente (ahora solo 0-4).
        """
        if self.game_over:
            return self._get_observation(self.current_player_idx), 0, True, False, {}

        try: action = MusAction(action_value)
        except ValueError: action = None
        
        player_id = self.current_player_idx
        legal_actions = self._get_legal_actions()

        if action not in legal_actions:
            reward = -1.0 # Castigo por acción ilegal
            
            if not legal_actions:
                # Esto es un estado de error, el juego no debería bloquearse
                obs = self._get_observation(player_id)
                return obs, reward, True, False, {"error": "Estado bloqueado sin acciones"} 
            
            # Corregir la acción a una legal aleatoria
            action = random.choice(legal_actions) 
        
        else:
            reward = 0
        
        # Guardar acción en el historial NUMÉRICO
        self.historial_acciones.append(action.value)

        # Procesar la FSM (¡MODIFICADO! 'ANNOUNCE' eliminado)
        if self.lance_state == "START":
            self._handle_action_start(action)
        elif self.lance_state == "BET_MADE":
            reward += self._handle_action_response(action)
            
        # Preparar el retorno estándar de Gym
        terminated = self.game_over
        truncated = False
        observation = self._get_observation(self.current_player_idx)
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        # (Sin cambios)
        pass

    def close(self):
        # (Sin cambios)
        pass