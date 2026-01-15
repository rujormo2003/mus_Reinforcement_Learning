import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from .mus_enums import MusAction, MusLance, LanceState
from .mus_entities import Baraja, Mano
from .mus_observation import MusObservationBuilder
from . import mus_rules

class MusEnvironment(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4} 

    def __init__(self, target_score=30):
        super().__init__() 
        
        self.target_score = target_score
        self.baraja = Baraja()
        
        self.scores = {0: 0, 1: 0}
        self.mano_player_idx = 0
        self.game_over = False
        
        self.manos: dict[int, Mano | None] = {0: None, 1: None}
        self.historial_acciones = [] 
        self.current_player_idx = 0
        self.current_lance: MusLance | None = None
        
        self.lance_bet = 0
        self.last_valid_bet = 0
        self.player_who_bet: int | None = None
        self.current_lance_state = LanceState.START 
        self.lances_list = list(MusLance)
        
        self.pending_bets = {l: 0 for l in MusLance}
        self.phase_winners = {l: None for l in MusLance} 
        
        self.observation_builder = MusObservationBuilder
        self.observation_space_size = self.observation_builder.get_observation_space_size()
        
        self.action_space = spaces.Discrete(len(MusAction))
        max_obs_value = max(40, self.target_score)
        self.observation_space = spaces.Box(
            low=-1.0, high=max_obs_value, 
            shape=(self.observation_space_size,), dtype=np.float32
        )
        
    def _resolve_lance_winner(self, lance=None) -> int:
        l = lance if lance else self.current_lance
        mano0, mano1 = self.manos[0], self.manos[1]
        mano_idx = self.mano_player_idx
        
        if l == MusLance.GRANDE: return mus_rules.compare_grande(mano0, mano1, mano_idx)
        if l == MusLance.CHICA: return mus_rules.compare_chica(mano0, mano1, mano_idx)
        if l == MusLance.PARES: return mus_rules.compare_pares(mano0, mano1, mano_idx)
        if l == MusLance.JUEGO: return mus_rules.compare_juego(mano0, mano1, mano_idx)
        return 0

    def _set_next_lance(self, lance: MusLance):
        self.current_lance = lance
        self.current_player_idx = self.mano_player_idx
        self.lance_bet = 0
        self.last_valid_bet = 0
        self.player_who_bet = None
        self.current_lance_state = LanceState.START

    def _auto_resolve_lance(self, lance_name: MusLance) -> bool:
        p0_has, p1_has = False, False
        if lance_name == MusLance.PARES:
            p0_has, p1_has = self.manos[0].tiene_pares(), self.manos[1].tiene_pares()
        elif lance_name == MusLance.JUEGO:
            p0_has, p1_has = self.manos[0].tiene_juego(), self.manos[1].tiene_juego()
        else:
            return False

        if p0_has and p1_has:
            return False 
        elif p0_has:
            self.scores[0] += 1 
        elif p1_has:
            self.scores[1] += 1
            
        self.pending_bets[lance_name] = -2 
        
        if p0_has != p1_has:
            self._check_game_over()
        return True

    def _move_to_next_lance(self):
        if self.game_over: return

        current_idx = -1
        if self.current_lance in self.lances_list:
            current_idx = self.lances_list.index(self.current_lance)
        
        next_lance_idx = current_idx + 1

        while next_lance_idx < len(self.lances_list):
            next_lance = self.lances_list[next_lance_idx]
            if not self._auto_resolve_lance(next_lance):
                self._set_next_lance(next_lance)
                return
            if self.game_over: return
            next_lance_idx += 1
        
        self._resolve_showdown()
        self._start_new_hand()

    def _resolve_showdown(self):
        if self.game_over: return
        
        for lance in self.lances_list:
            bet = self.pending_bets[lance]
            
            if bet == -2:
                ganador_idx = None
                if lance == MusLance.PARES:
                    if self.manos[0].tiene_pares(): ganador_idx = 0
                    elif self.manos[1].tiene_pares(): ganador_idx = 1
                elif lance == MusLance.JUEGO:
                    if self.manos[0].tiene_juego(): ganador_idx = 0
                    elif self.manos[1].tiene_juego(): ganador_idx = 1
                
                if ganador_idx is not None:
                    mano_g = self.manos[ganador_idx]
                    if lance == MusLance.PARES:
                        self.scores[ganador_idx] += self._get_pares_points(mano_g)
                    elif lance == MusLance.JUEGO:
                        self.scores[ganador_idx] += self._get_juego_points(mano_g)
                continue
            
            if bet == -1: continue

            forced_winner = self.phase_winners[lance]
            if forced_winner is not None:
                winner_idx = forced_winner
            else:
                winner_idx = self._resolve_lance_winner(lance)
            
            mano_ganadora = self.manos[winner_idx]
            
            if bet > 0:
                self.scores[winner_idx] += bet
            elif bet == 0 and forced_winner is None: 
                if lance in [MusLance.GRANDE, MusLance.CHICA]:
                    self.scores[winner_idx] += 1

            if lance == MusLance.PARES:
                self.scores[winner_idx] += self._get_pares_points(mano_ganadora)
            elif lance == MusLance.JUEGO:
                self.scores[winner_idx] += self._get_juego_points(mano_ganadora)
                
        self._check_game_over()

    def _get_pares_points(self, mano):
        if not mano.tiene_pares(): return 0
        info = mano.get_pares_info()
        t = info['tipo']
        if t == 'Pares': return 1
        if t == 'Medias': return 2
        if t == 'Duples': return 3
        return 0

    def _get_juego_points(self, mano):
        if mano.tiene_juego():
            v = mano.get_juego_valor()
            return 3 if v == 31 else 2
        return 1

    def _check_game_over(self):
        if self.scores[0] >= self.target_score or self.scores[1] >= self.target_score:
            self.game_over = True

    def _get_next_player(self) -> int: 
        return 1 - self.current_player_idx

    def _handle_action_start(self, action: MusAction):
        player = self.current_player_idx
        if action == MusAction.PASO:
            if self.current_player_idx == self.mano_player_idx:
                self.current_player_idx = self._get_next_player()
            else:
                self.pending_bets[self.current_lance] = 0
                self._move_to_next_lance()
        elif action == MusAction.ENVIDO:
            self.current_lance_state = LanceState.BET_MADE
            self.lance_bet = 2
            self.last_valid_bet = 0
            self.player_who_bet = player
            self.current_player_idx = self._get_next_player()
        elif action == MusAction.ORDAGO:
            self.current_lance_state = LanceState.BET_MADE
            self.lance_bet = self.target_score
            self.last_valid_bet = 0
            self.player_who_bet = player
            self.current_player_idx = self._get_next_player()
            
    def _handle_action_response(self, action: MusAction) -> dict:
        responder = self.current_player_idx
        better = self.player_who_bet
        info_extra = {}

        if action == MusAction.QUIERO:
            if self.lance_bet >= self.target_score: 
                winner = self._resolve_lance_winner()
                self.scores[winner] = self.target_score
                self._check_game_over()
            else:
                self.pending_bets[self.current_lance] = self.lance_bet
                self._move_to_next_lance()
            
        elif action == MusAction.NO_QUIERO:
            points_won = 1
            if self.lance_bet >= self.target_score:
                points_won = self.last_valid_bet if self.last_valid_bet > 0 else 1
            else:
                points_won = self.last_valid_bet if self.last_valid_bet > 0 else 1
            
            self.scores[better] += points_won
            info_extra = {"renuncia_winner": better, "renuncia_points": points_won}

            self.phase_winners[self.current_lance] = better
            self.pending_bets[self.current_lance] = -1 
            
            self._check_game_over()
            self._move_to_next_lance()
            
        elif action == MusAction.ORDAGO:
            if self.lance_bet == self.target_score: 
                self._handle_action_response(MusAction.QUIERO)
            else: 
                self.last_valid_bet = self.lance_bet
                self.current_lance_state = LanceState.BET_MADE
                self.lance_bet = self.target_score
                self.player_who_bet = responder
                self.current_player_idx = better
        
        return info_extra

    def _start_new_hand(self, is_first_hand: bool = False):
        if self.game_over: return

        if not is_first_hand:
            self.mano_player_idx = 1 - self.mano_player_idx

        self.baraja.crear_baraja(); self.baraja.mezclar()
        self.manos[0] = Mano(self.baraja.repartir(4))
        self.manos[1] = Mano(self.baraja.repartir(4))
        self.historial_acciones = [] 
        
        self.pending_bets = {l: 0 for l in MusLance}
        self.phase_winners = {l: None for l in MusLance}
        
        self.current_lance = None
        self._move_to_next_lance()

    def _get_legal_actions(self) -> list[MusAction]:
        if self.current_lance_state == LanceState.START:
            return [MusAction.PASO, MusAction.ENVIDO, MusAction.ORDAGO]
        if self.current_lance_state == LanceState.BET_MADE:
            if self.lance_bet >= self.target_score:
                return [MusAction.QUIERO, MusAction.NO_QUIERO]
            else:
                return [MusAction.QUIERO, MusAction.NO_QUIERO, MusAction.ORDAGO]
        return []

    def _get_observation(self, player_idx: int) -> np.ndarray:
        return self.observation_builder.build_observation(self, player_idx)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 
        self.scores = {0: 0, 1: 0}
        self.mano_player_idx = random.choice([0, 1])
        self.game_over = False
        self._start_new_hand(is_first_hand=True) 
        return self._get_observation(self.current_player_idx), {}

    def step(self, action_value: int):
        if self.game_over: return self._get_observation(0), 0, True, False, {}
        try: action = MusAction(action_value)
        except: action = random.choice(self._get_legal_actions())

        if action not in self._get_legal_actions(): action = random.choice(self._get_legal_actions())
        
        self.historial_acciones.append(action.value)
        
        info = {
            "action_name": action.name,
            "lance_prev": self.current_lance.name if self.current_lance else "None"
        }

        if self.current_lance_state == LanceState.START:
            self._handle_action_start(action)
        elif self.current_lance_state == LanceState.BET_MADE:
            extra = self._handle_action_response(action)
            if extra: info.update(extra)
            
        terminated = self.game_over
        info["lance_post"] = self.current_lance.name if self.current_lance else "None"
        
        return self._get_observation(self.current_player_idx), 0, terminated, False, info