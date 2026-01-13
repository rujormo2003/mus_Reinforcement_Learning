import os
import sys
import pygame
import time
from enum import Enum
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import random
import numpy as np

# --- IMPORTACIONES ---
try:
    from mus_gym.mus_env import MusEnvironment
    from mus_gym.mus_enums import MusAction, MusLance, LanceState
    from mus_gym.mus_entities import Carta, Mano
    from mus_gym.mus_config import PALOS, RANGOS_REALES
except ImportError:
    print("Error crítico: No se encuentra el paquete 'mus_gym'. Ejecuta desde la raíz.")
    sys.exit(1)

# --- CONFIGURACIÓN ---
pygame.init()
pygame.font.init()

# Colores
COLOR_VERDE_FONDO = (0, 100, 25)
COLOR_PANEL_FONDO = (30, 60, 30)
COLOR_BLANCO = (255, 255, 255)
COLOR_NEGRO = (0, 0, 0)
COLOR_AMARILLO = (255, 255, 0)
COLOR_ROJO_CLARO = (255, 150, 150)
COLOR_VERDE_CLARO = (150, 255, 150)
COLOR_LOG_IA = (255, 180, 180)
COLOR_LOG_HUMANO = (180, 255, 180)
COLOR_LOG_SISTEMA = (200, 200, 255)
COLOR_BOTON = (50, 50, 150)
COLOR_BOTON_HOVER = (80, 80, 180)
COLOR_BOTON_DES = (100, 100, 100)
COLOR_GRIS_CLARO = (200, 200, 200) # Añadido por si faltaba

ANCHO_JUEGO, ALTO_JUEGO = 900, 750
ANCHO_PANEL = 350
ANCHO_TOTAL = ANCHO_JUEGO + ANCHO_PANEL
ALTO_TOTAL = ALTO_JUEGO

# Fuentes
FONT_GRANDE = pygame.font.SysFont('Arial', 32, bold=True)
FONT_MEDIANA = pygame.font.SysFont('Arial', 24)
FONT_PEQUENA = pygame.font.SysFont('Arial', 18)
FONT_LOG = pygame.font.SysFont('Consolas', 15)

MODEL_PATH = "models/sheriff_colab/model_Exp_A_Baseline.zip"
ASSET_PATH = os.path.join("assets", "cartas")
CARD_SIZE = (85, 130)

class GameState(Enum):
    LOADING = 0
    HUMAN_TURN = 1
    AI_THINKING = 2
    REVEAL_HAND = 3
    GAME_OVER = 4

class MusRenderer:
    def __init__(self, pantalla):
        self.pantalla = pantalla
        self.card_images = {}
        self.dorso_img = None
        self.button_rects = {}
        self.load_assets()

    def load_assets(self):
        print(f"Cargando assets desde: {ASSET_PATH}")
        if not os.path.exists(ASSET_PATH):
            return

        try:
            for palo in PALOS:
                for rango in RANGOS_REALES:
                    key = f"{rango}_{palo}"
                    fname = f"{key}.png".lower()
                    path = os.path.join(ASSET_PATH, fname)
                    if os.path.exists(path):
                        img = pygame.image.load(path).convert_alpha()
                        self.card_images[key] = pygame.transform.scale(img, CARD_SIZE)
            
            p_dorso = os.path.join(ASSET_PATH, "dorso.png")
            if os.path.exists(p_dorso):
                self.dorso_img = pygame.image.load(p_dorso).convert_alpha()
                self.dorso_img = pygame.transform.scale(self.dorso_img, CARD_SIZE)
        except Exception as e:
            print(f"Error cargando imágenes: {e}")

    def draw_card(self, carta_obj, x, y, face_down=False):
        if face_down:
            if self.dorso_img:
                self.pantalla.blit(self.dorso_img, (x, y))
            else:
                pygame.draw.rect(self.pantalla, (0, 0, 150), (x, y, CARD_SIZE[0], CARD_SIZE[1]))
                pygame.draw.rect(self.pantalla, COLOR_BLANCO, (x, y, CARD_SIZE[0], CARD_SIZE[1]), 2)
        else:
            if carta_obj:
                key = f"{carta_obj.rango}_{carta_obj.palo}"
                if key in self.card_images:
                    self.pantalla.blit(self.card_images[key], (x, y))
                else:
                    pygame.draw.rect(self.pantalla, COLOR_BLANCO, (x, y, CARD_SIZE[0], CARD_SIZE[1]))
                    pygame.draw.rect(self.pantalla, COLOR_NEGRO, (x, y, CARD_SIZE[0], CARD_SIZE[1]), 2)
                    self.draw_text(f"{carta_obj.rango}", FONT_PEQUENA, COLOR_NEGRO, x+5, y+5)
                    self.draw_text(f"{carta_obj.palo[0]}", FONT_PEQUENA, COLOR_NEGRO, x+5, y+25)

    def draw_text(self, texto, font, color, x, y, center=False):
        surf = font.render(texto, True, color)
        rect = surf.get_rect()
        if center: rect.center = (x, y)
        else: rect.topleft = (x, y)
        self.pantalla.blit(surf, rect)

    def draw_game_state(self, game):
        self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_JUEGO, ALTO_TOTAL))
        self.pantalla.fill(COLOR_PANEL_FONDO, (ANCHO_JUEGO, 0, ANCHO_PANEL, ALTO_TOTAL))
        pygame.draw.line(self.pantalla, COLOR_BLANCO, (ANCHO_JUEGO, 0), (ANCHO_JUEGO, ALTO_TOTAL), 2)

        for i in range(4):
            self.draw_card(None, 250 + i * 110, 30, face_down=True)
        
        mano = game.env.manos[0]
        if mano:
            for i, c in enumerate(mano.cartas):
                self.draw_card(c, 250 + i * 110, ALTO_JUEGO - 160)
        
        self._draw_center_info(game)
        self._draw_buttons(game)
        self._draw_panel(game)

    def _draw_center_info(self, game):
        if game.state == GameState.HUMAN_TURN:
            txt, col = "¡TU TURNO!", COLOR_AMARILLO
        elif game.state == GameState.AI_THINKING:
            txt, col = "IA PENSANDO...", COLOR_GRIS_CLARO
        else:
            txt, col = "...", COLOR_GRIS_CLARO
        self.draw_text(txt, FONT_GRANDE, col, ANCHO_JUEGO//2, 280, center=True)
        
        lance = game.env.current_lance.name if game.env.current_lance else "N/A"
        self.draw_text(f"FASE: {lance}", FONT_GRANDE, COLOR_BLANCO, ANCHO_JUEGO//2, 330, center=True)
        
        if game.env.lance_bet > 0:
            bet_txt = f"Hay apuesta: {game.env.lance_bet}"
            if game.env.lance_bet >= game.env.target_score: bet_txt = "¡¡ ÓRDAGO !!"
            self.draw_text(bet_txt, FONT_GRANDE, COLOR_ROJO_CLARO, ANCHO_JUEGO//2, 380, center=True)
            
        mano_txt = "Eres Mano" if game.env.mano_player_idx == 0 else "IA es Mano"
        self.draw_text(f"({mano_txt})", FONT_PEQUENA, COLOR_GRIS_CLARO, ANCHO_JUEGO//2, 240, center=True)

    def _draw_buttons(self, game):
        self.button_rects.clear()
        is_human = (game.state == GameState.HUMAN_TURN)
        legal = game.env._get_legal_actions() if is_human else []
        
        w, h = 140, 50
        start_x = (ANCHO_JUEGO - (len(legal)*(w+10))) // 2
        mouse = pygame.mouse.get_pos()
        
        for i, act in enumerate(legal):
            r = pygame.Rect(start_x + i*(w+10), 450, w, h)
            self.button_rects[act] = r
            
            col = COLOR_BOTON_HOVER if (is_human and r.collidepoint(mouse)) else COLOR_BOTON
            if not is_human: col = COLOR_BOTON_DES
            
            pygame.draw.rect(self.pantalla, col, r, border_radius=8)
            pygame.draw.rect(self.pantalla, COLOR_BLANCO, r, 2, border_radius=8)
            self.draw_text(act.name, FONT_MEDIANA, COLOR_BLANCO, r.centerx, r.centery, center=True)

    def _draw_panel(self, game):
        x = ANCHO_JUEGO + 20
        self.draw_text(f"TU: {game.env.scores[0]}", FONT_GRANDE, COLOR_VERDE_CLARO, x, 30)
        self.draw_text(f"IA: {game.env.scores[1]}", FONT_GRANDE, COLOR_ROJO_CLARO, x+180, 30)
        pygame.draw.line(self.pantalla, COLOR_GRIS_CLARO, (x, 70), (ANCHO_TOTAL-20, 70))
        
        m = game.env.manos[0]
        if m:
            self.draw_text("TU MANO:", FONT_MEDIANA, COLOR_AMARILLO, x, 90)
            p = m.get_pares_info()
            j = m.get_juego_valor()
            self.draw_text(f"- {p['tipo']}", FONT_PEQUENA, COLOR_BLANCO, x+10, 120)
            self.draw_text(f"- Juego: {j} ({'Sí' if m.tiene_juego() else 'No'})", FONT_PEQUENA, COLOR_BLANCO, x+10, 145)

        self.draw_text("HISTORIAL:", FONT_MEDIANA, COLOR_AMARILLO, x, 250)
        y_log = 290
        for msg, col in game.message_log[-22:]:
            self.draw_text(msg, FONT_LOG, col, x+5, y_log)
            y_log += 18

    def draw_reveal(self, game):
        self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_TOTAL, ALTO_TOTAL))
        self.draw_text("--- FIN DE LA MANO ---", FONT_GRANDE, COLOR_AMARILLO, ANCHO_TOTAL//2, 50, center=True)
        self.draw_text("[ESPACIO] para continuar", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL//2, 90, center=True)
        
        self._draw_final_hands(game)

        self.draw_text(f"PUNTUACIÓN ACTUAL: TU {game.env.scores[0]} - IA {game.env.scores[1]}", FONT_GRANDE, COLOR_BLANCO, ANCHO_TOTAL//2, 620, center=True)

    def draw_game_over(self, game):
        # CAMBIO: Ahora usamos el mismo fondo verde y mostramos las cartas finales
        self.pantalla.fill(COLOR_VERDE_FONDO, (0,0,ANCHO_TOTAL, ALTO_TOTAL))
        
        ganador = "¡HAS GANADO LA PARTIDA!" if game.env.scores[0] >= 30 else "LA IA HA GANADO LA PARTIDA"
        c = COLOR_AMARILLO if game.env.scores[0] >= 30 else COLOR_ROJO_CLARO
        
        self.draw_text(ganador, FONT_GRANDE, c, ANCHO_TOTAL//2, 50, center=True)
        self.draw_text(f"MARCADOR FINAL: TU {game.env.scores[0]} - IA {game.env.scores[1]}", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL//2, 90, center=True)
        self.draw_text("Pulsa [ESPACIO] para SALIR", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL//2, 130, center=True)

        # ¡AQUÍ ESTÁ LA MAGIA! Dibujamos las cartas de la última mano
        self._draw_final_hands(game)

    def _draw_final_hands(self, game):
        # Helper para dibujar manos finales (snapshot)
        h_ia = game.last_hands_snapshot[1]
        if h_ia:
            self.draw_text("IA tenía:", FONT_MEDIANA, COLOR_ROJO_CLARO, ANCHO_TOTAL//2, 170, center=True)
            for i, c in enumerate(h_ia.cartas):
                self.draw_card(c, 250 + i*110, 210)
            info = f"{h_ia.get_pares_info()['tipo']} | Juego: {h_ia.get_juego_valor()}"
            self.draw_text(info, FONT_PEQUENA, COLOR_GRIS_CLARO, ANCHO_TOTAL//2, 350, center=True)

        h_pl = game.last_hands_snapshot[0]
        if h_pl:
            self.draw_text("Tú tenías:", FONT_MEDIANA, COLOR_VERDE_CLARO, ANCHO_TOTAL//2, 400, center=True)
            for i, c in enumerate(h_pl.cartas):
                self.draw_card(c, 250 + i*110, 440)


class Game:
    def __init__(self):
        self.pantalla = pygame.display.set_mode((ANCHO_TOTAL, ALTO_TOTAL))
        pygame.display.set_caption("Mus Master AI")
        self.clock = pygame.time.Clock()
        self.renderer = MusRenderer(self.pantalla)
        
        self.env = None
        self.model = None
        self.running = True
        self.state = GameState.LOADING
        
        self.message_log = []
        self.last_hands_snapshot = (None, None)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

        self.init_env()

    def init_env(self):
        self.env = MusEnvironment(target_score=30)
        if os.path.exists(MODEL_PATH):
            self.model = RecurrentPPO.load(MODEL_PATH)
            self.log("Modelo IA cargado.", COLOR_LOG_SISTEMA)
        else:
            self.log("Modelo no encontrado. Jugando Random.", COLOR_LOG_IA)
        
        self.env.reset()
        self.state = self.get_turn_state()

    def log(self, msg, color):
        t = time.strftime("%H:%M:%S")
        self.message_log.append((f"[{t}] {msg}", color))
        print(f"LOG [{t}]: {msg}")

    def get_turn_state(self):
        return GameState.HUMAN_TURN if self.env.current_player_idx == 0 else GameState.AI_THINKING

    def run(self):
        while self.running:
            action = self.handle_events()
            
            if self.state == GameState.HUMAN_TURN and action and isinstance(action, MusAction):
                self.step(action)
            elif self.state == GameState.AI_THINKING:
                pygame.time.wait(600)
                self.ai_step()
            
            if self.state == GameState.REVEAL_HAND:
                self.renderer.draw_reveal(self)
            elif self.state == GameState.GAME_OVER:
                self.renderer.draw_game_over(self)
            else:
                self.renderer.draw_game_state(self)
                
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if self.state == GameState.REVEAL_HAND:
                    self.log("--- Nueva Mano ---", COLOR_LOG_SISTEMA)
                    self.state = self.get_turn_state()
                    return None
                if self.state == GameState.GAME_OVER:
                    self.running = False
                    return None

            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.state == GameState.HUMAN_TURN:
                    for act, rect in self.renderer.button_rects.items():
                        if rect.collidepoint(event.pos): return act
                elif self.state == GameState.REVEAL_HAND:
                     self.log("--- Nueva Mano ---", COLOR_LOG_SISTEMA)
                     self.state = self.get_turn_state()
                     return None
        return None

    def step(self, action: MusAction):
        scores_pre = self.env.scores.copy()
        lance_pre = self.env.current_lance
        
        # Guardamos snapshot para el REVEAL (deepcopy para asegurar integridad)
        import copy
        self.last_hands_snapshot = (copy.deepcopy(self.env.manos[0]), copy.deepcopy(self.env.manos[1]))

        actor = "TU" if self.env.current_player_idx == 0 else "IA"
        c = COLOR_LOG_HUMANO if actor == "TU" else COLOR_LOG_IA
        self.log(f"{actor}: {action.name}", c)

        obs, reward, term, trunc, info = self.env.step(action.value)
        self.episode_starts[0] = term or trunc

        # --- LOGS INTELIGENTES ---
        
        # 1. ¿Hubo renuncia?
        renuncia_winner = info.get("renuncia_winner")
        renuncia_pts = info.get("renuncia_points", 0)
        
        if renuncia_winner is not None:
            if renuncia_winner == 0:
                self.log(f"-> Ganas {renuncia_pts} por renuncia", COLOR_VERDE_CLARO)
            else:
                self.log(f"-> IA gana {renuncia_pts} por renuncia", COLOR_ROJO_CLARO)

        # 2. Puntos del resto (Recuento o Paso inmediato)
        diff_tu = self.env.scores[0] - scores_pre[0]
        diff_ia = self.env.scores[1] - scores_pre[1]
        
        if renuncia_winner == 0: diff_tu -= renuncia_pts
        if renuncia_winner == 1: diff_ia -= renuncia_pts

        # 3. Detectar fin de mano
        lance_post = self.env.current_lance
        hand_ended = False
        if lance_post and lance_pre:
             if lance_post.value < lance_pre.value and not term:
                 hand_ended = True

        if term:
            self.state = GameState.GAME_OVER
        elif hand_ended:
            self.state = GameState.REVEAL_HAND
            # Logs de recuento final
            if diff_tu > 0: self.log(f"RECUENTO: +{diff_tu} para ti", COLOR_AMARILLO)
            if diff_ia > 0: self.log(f"RECUENTO: +{diff_ia} para IA", COLOR_AMARILLO)
        else:
            # Puntos intermedios (ej: punto de paso)
            if diff_tu > 0: self.log(f"-> Ganas {diff_tu} pto(s)", COLOR_VERDE_CLARO)
            if diff_ia > 0: self.log(f"-> IA gana {diff_ia} pto(s)", COLOR_ROJO_CLARO)
            
            if lance_post != lance_pre:
                self.log(f"Cambio a: {lance_post.name}", COLOR_LOG_SISTEMA)
            self.state = self.get_turn_state()

    def ai_step(self):
        obs = self.env._get_observation(1)
        if self.model:
            act_val, self.lstm_states = self.model.predict(
                obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=False
            )
            val = int(act_val)
        else:
            legal = self.env._get_legal_actions()
            val = random.choice(legal).value
            
        legal_actions = self.env._get_legal_actions()
        try: act = MusAction(val)
        except: act = random.choice(legal_actions)

        if act not in legal_actions: act = random.choice(legal_actions)
        self.step(act)

if __name__ == "__main__":
    Game().run()