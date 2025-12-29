# import os
# import sys
# import pygame
# import time
# from enum import Enum
# from stable_baselines3 import PPO
# import random
# import numpy as np

# # Importamos el entorno y las clases.
# try:
#     # (Asumiendo que este script está en el directorio raíz,
#     # y 'mus_gym' es un subdirectorio)
#     from mus_gym.mus_env import MusEnvironment
#     from mus_gym.mus_enums import MusAction, MusLance, LanceState
#     from mus_gym.mus_entities import Carta, Mano
#     from mus_gym.mus_config import PALOS, RANGOS_REALES
# except ImportError:
#     print("Error: No se pudo encontrar el paquete 'mus_gym'.")
#     print("Asegúrate de que el script está en el directorio correcto (fuera de 'mus_gym')")
#     print("y que 'mus_gym' es un paquete (contiene __init__.py)")
#     sys.exit(1)

# # --- 1. CONFIGURACIÓN Y CONSTANTES ---
# pygame.init()
# pygame.font.init()

# # Colores
# COLOR_BLANCO = (255, 255, 255)
# COLOR_NEGRO = (0, 0, 0)
# COLOR_GRIS_CLARO = (200, 200, 200)
# COLOR_VERDE_FONDO = (0, 100, 25)
# COLOR_PANEL_FONDO = (30, 60, 30)
# COLOR_BOTON = (50, 50, 150)
# COLOR_BOTON_HOVER = (80, 80, 180)
# COLOR_BOTON_DESHABILITADO = (100, 100, 100)
# COLOR_AMARILLO = (255, 255, 0)
# COLOR_ROJO_CLARO = (255, 150, 150)
# COLOR_LOG_IA = (255, 180, 180)
# COLOR_LOG_HUMANO = (180, 255, 180)
# COLOR_LOG_SISTEMA = (200, 200, 255)

# # Pantalla
# ANCHO_JUEGO, ALTO_JUEGO = 800, 700
# ANCHO_PANEL = 300
# ANCHO_TOTAL = ANCHO_JUEGO + ANCHO_PANEL
# ALTO_TOTAL = ALTO_JUEGO

# # Fuentes
# try:
#     FONT_GRANDE = pygame.font.SysFont('Arial', 32)
#     FONT_MEDIANA = pygame.font.SysFont('Arial', 24)
#     FONT_PEQUENA = pygame.font.SysFont('Arial', 18)
#     FONT_LOG = pygame.font.SysFont('Consolas', 16)
# except pygame.error:
#     FONT_GRANDE = pygame.font.Font(None, 40)
#     FONT_MEDIANA = pygame.font.Font(None, 30)
#     FONT_PEQUENA = pygame.font.Font(None, 24)
#     FONT_LOG = pygame.font.Font(None, 22)

# # Ruta del modelo
# MODEL_PATH = "models/sheriff_colab/FINAL_MUS_MODEL.zip" # <-- CAMBIA ESTO A LA RUTA DE TU MODELO
# ASSET_PATH = os.path.join("assets", "cartas")
# CARD_SIZE = (80, 120)

# class GameState(Enum):
#     """ Máquina de estados para controlar el flujo del juego """
#     LOADING = 0
#     HUMAN_TURN = 1
#     AI_THINKING = 2
#     AI_STEP = 3
#     REVEAL_HAND = 4
#     GAME_OVER = 5

# # --- 2. CLASE MusRenderer ---

# class MusRenderer:
#     """ Gestiona toda la carga de assets y la lógica de dibujado. """
#     def __init__(self, pantalla):
#         self.pantalla = pantalla
#         self.card_images = {}
#         self.dorso_img = None
#         self.button_rects = {}
#         self.load_assets()

#     def load_assets(self):
#         print("Cargando imágenes de cartas...")
#         try:
#             for palo in PALOS:
#                 for rango in RANGOS_REALES:
#                     key = f"{rango}_{palo}"
#                     filename = f"{key}.png".lower()
#                     path = os.path.join(ASSET_PATH, filename)
#                     image = pygame.image.load(path).convert_alpha()
#                     self.card_images[key] = pygame.transform.scale(image, CARD_SIZE)
            
#             path = os.path.join(ASSET_PATH, "dorso.png")
#             self.dorso_img = pygame.image.load(path).convert_alpha()
#             self.dorso_img = pygame.transform.scale(self.dorso_img, CARD_SIZE)
#             print("¡Imágenes cargadas con éxito!")
#         except Exception as e:
#             print(f"--- ¡ERROR DE ASSETS! ---")
#             print(f"No se pudo cargar una imagen: {e}")
#             print(f"Asegúrate de que la carpeta '{ASSET_PATH}' existe y contiene las imágenes.")
#             pygame.quit()
#             sys.exit()

#     def draw_card(self, carta_obj, x, y, face_down=False):
#         if face_down or not carta_obj:
#             self.pantalla.blit(self.dorso_img, (x, y))
#         else:
#             key = f"{carta_obj.rango}_{carta_obj.palo}"
#             if key in self.card_images:
#                 self.pantalla.blit(self.card_images[key], (x, y))
#             else:
#                 pygame.draw.rect(self.pantalla, (255, 0, 0), (x, y, CARD_SIZE[0], CARD_SIZE[1]))

#     def draw_text(self, texto, font, color, x, y, center=False):
#         text_surface = font.render(texto, True, color)
#         text_rect = text_surface.get_rect()
#         if center:
#             text_rect.center = (x, y)
#         else:
#             text_rect.topleft = (x, y)
#         self.pantalla.blit(text_surface, text_rect)

#     def draw_game_state(self, game):
#         """ Dibuja el estado principal del juego (tablero y panel) """
        
#         # 1. Fondo del tablero
#         self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_JUEGO, ALTO_TOTAL))
#         # 2. Fondo del panel
#         self.pantalla.fill(COLOR_PANEL_FONDO, (ANCHO_JUEGO, 0, ANCHO_PANEL, ALTO_TOTAL))
#         pygame.draw.line(self.pantalla, COLOR_BLANCO, (ANCHO_JUEGO, 0), (ANCHO_JUEGO, ALTO_TOTAL), 2)

#         # 3. Dibujar Manos
#         mano_humano = game.env.manos[0]
#         mano_ia = game.env.manos[1]
        
#         # Mano IA (Arriba)
#         for i in range(4):
#             self.draw_card(None, 200 + i * 100, 30, face_down=True)
        
#         # Mano Humano (Abajo)
#         if mano_humano:
#             for i, carta in enumerate(mano_humano.cartas):
#                 self.draw_card(carta, 200 + i * 100, ALTO_JUEGO - 150)
        
#         # 4. Dibujar Indicador de Turno y Estado
#         self.draw_turn_indicator(game)

#         # 5. Dibujar Botones de Acción
#         self.draw_action_buttons(game)

#         # 6. Dibujar Panel de Información
#         self.draw_info_panel(game)

#     def draw_turn_indicator(self, game):
#         """ Dibuja el estado actual del juego en el centro """
#         if game.state == GameState.HUMAN_TURN:
#             texto_turno = "¡ES TU TURNO!"
#             color_turno = COLOR_AMARILLO
#         elif game.state == GameState.AI_THINKING:
#             texto_turno = "IA PENSANDO..."
#             color_turno = COLOR_GRIS_CLARO
#         else:
#             texto_turno = "..."
#             color_turno = COLOR_GRIS_CLARO

#         self.draw_text(texto_turno, FONT_GRANDE, color_turno, ANCHO_JUEGO // 2, 250, center=True)
        
#         # Info del Lance
#         lance_txt = f"Lance: {game.env.current_lance.name if game.env.current_lance else 'N/A'}"
#         estado_txt = f"Estado: {game.env.current_lance_state.name}"
#         apuesta_txt = f"Apuesta: {game.env.lance_bet} puntos"

#         self.draw_text(lance_txt, FONT_MEDIANA, COLOR_BLANCO, ANCHO_JUEGO // 2, 300, center=True)
#         self.draw_text(estado_txt, FONT_MEDIANA, COLOR_BLANCO, ANCHO_JUEGO // 2, 330, center=True)
#         if game.env.lance_bet > 0:
#             self.draw_text(apuesta_txt, FONT_MEDIANA, COLOR_AMARILLO, ANCHO_JUEGO // 2, 360, center=True)
        
#         # Quién es "mano"
#         mano_txt = f"JUGADOR {game.env.mano_player_idx} ES 'MANO'"
#         self.draw_text(mano_txt, FONT_PEQUENA, COLOR_GRIS_CLARO, ANCHO_JUEGO // 2, 200, center=True)

#     def draw_action_buttons(self, game):
#         """ Dibuja los botones de acción para el humano """
#         self.button_rects.clear()
        
#         es_turno_humano = (game.state == GameState.HUMAN_TURN)
#         legal_actions = game.env._get_legal_actions() if es_turno_humano else []
        
#         ancho_boton = 150
#         alto_boton = 50
#         num_botones = len(legal_actions)
#         start_x = (ANCHO_JUEGO - (num_botones * (ancho_boton + 20) - 20)) // 2
#         mouse_pos = pygame.mouse.get_pos()
        
#         for i, action in enumerate(legal_actions):
#             button_rect = pygame.Rect(start_x + i * (ancho_boton + 20), 400, ancho_boton, alto_boton)
#             self.button_rects[action] = button_rect
            
#             color = COLOR_BOTON
#             if es_turno_humano:
#                 if button_rect.collidepoint(mouse_pos):
#                     color = COLOR_BOTON_HOVER
#             else:
#                 color = COLOR_BOTON_DESHABILITADO
                
#             pygame.draw.rect(self.pantalla, color, button_rect, border_radius=10)
#             self.draw_text(action.name, FONT_MEDIANA, COLOR_BLANCO, button_rect.centerx, button_rect.centery, center=True)

#     def draw_info_panel(self, game):
#         """ Dibuja el panel lateral con puntuaciones, info de mano y log """
#         x_base = ANCHO_JUEGO + 20
        
#         # 1. Puntuaciones
#         self.draw_text("PUNTUACIÓN", FONT_GRANDE, COLOR_AMARILLO, x_base, 20)
#         self.draw_text(f"Humano: {game.env.scores[0]}", FONT_MEDIANA, COLOR_BLANCO, x_base + 10, 60)
#         self.draw_text(f"IA: {game.env.scores[1]}", FONT_MEDIANA, COLOR_BLANCO, x_base + 10, 90)
        
#         # 2. Info de Mano
#         self.draw_text("INFO DE MANO", FONT_GRANDE, COLOR_AMARILLO, x_base, 150)
        
#         mano_humano = game.env.manos[0]
#         mano_ia = game.env.manos[1]
        
#         # Info Humano
#         self.draw_text("Humano:", FONT_MEDIANA, COLOR_BLANCO, x_base + 10, 190)
#         if mano_humano:
#             info_pares_h = mano_humano.get_pares_info()
#             pares_txt_h = f"Pares: {info_pares_h['tipo']}"
#             juego_txt_h = f"Juego: {'Sí' if mano_humano.tiene_juego() else 'No'} ({mano_humano.get_juego_valor()})"
#             self.draw_text(pares_txt_h, FONT_PEQUENA, COLOR_GRIS_CLARO, x_base + 20, 220)
#             self.draw_text(juego_txt_h, FONT_PEQUENA, COLOR_GRIS_CLARO, x_base + 20, 240)
        
#         # Info IA (Oculta)
#         self.draw_text("IA:", FONT_MEDIANA, COLOR_BLANCO, x_base + 10, 280)
#         if mano_ia:
#             self.draw_text("Pares: ???", FONT_PEQUENA, COLOR_GRIS_CLARO, x_base + 20, 310)
#             self.draw_text("Juego: ???", FONT_PEQUENA, COLOR_GRIS_CLARO, x_base + 20, 330)
            
#         # 3. Log de Acciones
#         self.draw_text("LOG DE ACCIONES", FONT_GRANDE, COLOR_AMARILLO, x_base, 400)
#         pygame.draw.rect(self.pantalla, COLOR_NEGRO, (x_base - 5, 435, ANCHO_PANEL - 30, ALTO_TOTAL - 450))
        
#         y_log = 440
#         max_logs = 12
#         # Mostrar los últimos 'max_logs' mensajes
#         for msg, color in game.message_log[-max_logs:]:
#             self.draw_text(msg, FONT_LOG, color, x_base, y_log)
#             y_log += FONT_LOG.get_height()

#     def draw_reveal_screen(self, game):
#         """ Dibuja la pantalla de fin de mano, mostrando toda la info """
#         self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_TOTAL, ALTO_TOTAL))
        
#         mano_humano = game.old_manos[0]
#         mano_ia = game.old_manos[1]

#         self.draw_text("FIN DE LA MANO", FONT_GRANDE, COLOR_AMARILLO, ANCHO_TOTAL // 2, 50, center=True)

#         # Cartas IA
#         self.draw_text("Mano de la IA:", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL // 2, 120, center=True)
#         if mano_ia:
#             for i, carta in enumerate(mano_ia.cartas):
#                 self.draw_card(carta, 200 + i * 100, 160, face_down=False)
            
#             info_pares_ia = mano_ia.get_pares_info()
#             pares_txt_ia = f"Pares: {info_pares_ia['tipo']} ({info_pares_ia['valor_principal']})"
#             juego_txt_ia = f"Juego: {'Sí' if mano_ia.tiene_juego() else 'No'} ({mano_ia.get_juego_valor()})"
#             self.draw_text(pares_txt_ia, FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, 300, center=True)
#             self.draw_text(juego_txt_ia, FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, 330, center=True)

#         # Cartas Humano
#         self.draw_text("Tu Mano:", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL // 2, 400, center=True)
#         if mano_humano:
#             for i, carta in enumerate(mano_humano.cartas):
#                 self.draw_card(carta, 200 + i * 100, 440, face_down=False)

#             info_pares_h = mano_humano.get_pares_info()
#             pares_txt_h = f"Pares: {info_pares_h['tipo']} ({info_pares_h['valor_principal']})"
#             juego_txt_h = f"Juego: {'Sí' if mano_humano.tiene_juego() else 'No'} ({mano_humano.get_juego_valor()})"
#             self.draw_text(pares_txt_h, FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, 580, center=True)
#             self.draw_text(juego_txt_h, FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, 610, center=True)

#         self.draw_text("Pulsa [ESPACIO] para continuar...", FONT_MEDIANA, COLOR_AMARILLO, ANCHO_TOTAL // 2, 670, center=True)
#         pygame.display.flip()

#     def draw_game_over_screen(self, game):
#         """ Dibuja la pantalla de fin de partida """
#         self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_TOTAL, ALTO_TOTAL))
        
#         texto_final = "¡HAS GANADO!" if game.env.scores[0] > game.env.scores[1] else "HA GANADO LA IA"
#         self.draw_text(texto_final, FONT_GRANDE, COLOR_AMARILLO, ANCHO_TOTAL // 2, ALTO_TOTAL // 2 - 50, center=True)
        
#         score_txt = f"Puntuación Final: {game.env.scores[0]} - {game.env.scores[1]}"
#         self.draw_text(score_txt, FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL // 2, ALTO_TOTAL // 2, center=True)
        
#         self.draw_text("Haz clic o pulsa [ESPACIO] para salir", FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, ALTO_TOTAL // 2 + 50, center=True)
#         pygame.display.flip()

# # --- 3. CLASE Game (El motor) ---
# # --- 3. CLASE Game (El motor) ---

# class Game:
#     """
#     Gestiona el bucle principal del juego, el estado,
#     el entorno de Gym y el modelo de SB3.
#     """
#     def __init__(self):
#         self.pantalla = pygame.display.set_mode((ANCHO_TOTAL, ALTO_TOTAL))
#         pygame.display.set_caption("Mus: Humano vs IA (LSTM)")
#         self.clock = pygame.time.Clock()
#         self.renderer = MusRenderer(self.pantalla)
        
#         self.env = None
#         self.model = None
#         self.running = True
#         self.state = GameState.LOADING
        
#         # Variables de estado del juego
#         self.obs_vector = None
#         self.last_ia_action = None
#         self.old_manos = (None, None)
#         self.last_lance = None
        
#         # --- MEMORIA LSTM (CRUCIAL) ---
#         # Almacena los estados ocultos de la red neuronal entre pasos.
#         # Si no se pasan estos estados, la IA tendrá "amnesia".
#         self._last_lstm_states = None
#         self._last_episode_starts = np.ones((1,), dtype=bool)
        
#         self.message_log = []
#         self.ai_think_timer = 0
        
#         self.load_model_and_env()
#         self.add_log("¡Bienvenido a Mus vs IA!", COLOR_LOG_SISTEMA)

#     def load_model_and_env(self):
#         if not os.path.exists(MODEL_PATH):
#             print(f"Error: No se encontró el modelo en {MODEL_PATH}")
#             self.add_log(f"Error: Modelo no encontrado", COLOR_ROJO_CLARO)
#             self.running = False
#             return
            
#         try:
#             print(f"Cargando modelo IA desde {MODEL_PATH}...")
#             # Importamos RecurrentPPO aquí para asegurar que está disponible
#             from sb3_contrib import RecurrentPPO
#             self.model = RecurrentPPO.load(MODEL_PATH)
#             print("Modelo cargado.")
            
#             # Creamos el entorno base (sin bots, controlamos ambos lados)
#             self.env = MusEnvironment(target_score=30)
#             print("Entorno creado.")
            
#         except Exception as e:
#             print(f"Error al cargar modelo o entorno: {e}")
#             self.add_log(f"Error al cargar: {e}", COLOR_ROJO_CLARO)
#             self.running = False

#     def add_log(self, message, color):
#         """ Añade un mensaje al log en pantalla """
#         print(f"LOG: {message}") 
#         self.message_log.append((message, color))
#         if len(self.message_log) > 20: 
#             self.message_log.pop(0)

#     def run(self):
#         """ Bucle principal del juego """
#         if not self.running:
#             return 
            
#         # Reseteo inicial
#         self.obs_vector, info = self.env.reset(seed=random.randint(0, 10000))
        
#         # Reseteamos la memoria LSTM al iniciar la aplicación
#         self._last_lstm_states = None
#         self._last_episode_starts = np.ones((1,), dtype=bool)
        
#         self.state = self.get_state_from_env()
#         self.last_lance = self.env.current_lance

#         while self.running:
#             # 1. Manejar Eventos (Input)
#             action = self.handle_events()
            
#             # 2. Actualizar Estado del Juego (Lógica)
#             if self.state == GameState.HUMAN_TURN:
#                 if action:
#                     self.execute_step(action)
            
#             elif self.state == GameState.AI_THINKING:
#                 if pygame.time.get_ticks() > self.ai_think_timer:
#                     self.execute_ai_step()
            
#             elif self.state == GameState.REVEAL_HAND:
#                 if action == "CONTINUE":
#                     # Avanzamos al estado real del juego (nueva mano)
#                     self.state = self.get_state_from_env()
#                     self.add_log("--- Nueva Mano ---", COLOR_LOG_SISTEMA)
#                     # NOTA: No reseteamos _last_lstm_states aquí porque es la misma partida
#                     # y la IA debe recordar el contexto general.

#             elif self.state == GameState.GAME_OVER:
#                 if action: 
#                     self.running = False
            
#             # 3. Dibujar (Renderizado)
#             self.render()
            
#             # 4. Controlar FPS
#             self.clock.tick(30) 
            
#         pygame.quit()
#         sys.exit()

#     def handle_events(self):
#         """ Procesa inputs de Pygame """
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 self.running = False
#                 return None
                
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
#                     return "CONTINUE" 

#             if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
#                 if self.state == GameState.GAME_OVER or self.state == GameState.REVEAL_HAND:
#                     return "CONTINUE"
                    
#                 if self.state == GameState.HUMAN_TURN:
#                     mouse_pos = event.pos
#                     for action, rect in self.renderer.button_rects.items():
#                         if rect.collidepoint(mouse_pos):
#                             return action 
#         return None

#     def execute_step(self, action: MusAction):
#         """ 
#         Ejecuta un paso en el entorno y actualiza el estado.
#         """
#         # Guardar estado ANTES del paso (para visualización)
#         self.old_manos = (self.env.manos[0], self.env.manos[1])
#         self.last_lance = self.env.current_lance
#         old_scores = self.env.scores.copy()
        
#         # Logging
#         es_humano = (self.env.current_player_idx == 0)
#         actor_name = "Humano" if es_humano else "IA"
#         actor_color = COLOR_LOG_HUMANO if es_humano else COLOR_LOG_IA
#         self.add_log(f"{actor_name}: {action.name}", actor_color)
        
#         if not es_humano:
#             self.last_ia_action = action

#         # --- PASO EN GYM ---
#         self.obs_vector, reward, terminated, truncated, info = self.env.step(action.value)
        
#         # Actualizar flag de episodio para la LSTM
#         # Si termina, el siguiente paso será un "inicio de episodio" para la memoria
#         self._last_episode_starts[0] = terminated or truncated

#         # --- Logs de puntuación ---
#         if self.env.scores[0] > old_scores[0]:
#             diff = self.env.scores[0] - old_scores[0]
#             self.add_log(f"Humano gana {diff} punto(s)", COLOR_LOG_SISTEMA)
#         if self.env.scores[1] > old_scores[1]:
#             diff = self.env.scores[1] - old_scores[1]
#             self.add_log(f"IA gana {diff} punto(s)", COLOR_LOG_SISTEMA)

#         # --- Máquina de Estados ---
#         if terminated or truncated:
#             self.state = GameState.GAME_OVER
#             self.add_log("¡¡¡PARTIDA TERMINADA!!!", COLOR_LOG_SISTEMA)
        
#         # Detectar fin de mano: Si volvemos a GRANDE desde un lance avanzado
#         elif self.env.current_lance == MusLance.GRANDE and self.last_lance in [MusLance.CHICA, MusLance.PARES, MusLance.JUEGO]:
#             self.state = GameState.REVEAL_HAND
#             self.add_log("Fin de la mano. Mostrando cartas...", COLOR_LOG_SISTEMA)
            
#         else:
#             self.state = self.get_state_from_env()

#     def execute_ai_step(self):
#         """ Pide acción a la IA gestionando su memoria LSTM """
#         self.state = GameState.AI_STEP 
        
#         # 1. Obtener observación explícita de la IA (Jugador 1)
#         obs_ia = self.env._get_observation(player_idx=1)
        
#         # 2. Predecir usando MEMORIA
#         # Pasamos state (memoria anterior) y episode_start
#         action_val, lstm_states = self.model.predict(
#             obs_ia, 
#             state=self._last_lstm_states, 
#             episode_start=self._last_episode_starts,
#             deterministic=False 
#         )
        
#         # 3. Actualizar la memoria para el futuro
#         self._last_lstm_states = lstm_states
        
#         # 4. Convertir acción
#         action_enum = MusAction(action_val.item())
        
#         # Validación anti-trampas
#         legal_actions = self.env._get_legal_actions()
#         if action_enum not in legal_actions:
#             self.add_log(f"IA intentó ilegal: {action_enum.name}. Corrigiendo...", COLOR_ROJO_CLARO)
#             action_enum = random.choice(legal_actions)
            
#         self.execute_step(action_enum)

#     def get_state_from_env(self):
#         """ Determina de quién es el turno """
#         if self.env.current_player_idx == 0:
#             return GameState.HUMAN_TURN
#         else:
#             # Pausa dramática para que parezca que piensa
#             self.ai_think_timer = pygame.time.get_ticks() + 1000 
#             return GameState.AI_THINKING

#     def render(self):
#         """ Delega el dibujo al renderer según el estado """
#         if self.state == GameState.GAME_OVER:
#             self.renderer.draw_game_over_screen(self)
#         elif self.state == GameState.REVEAL_HAND:
#             self.renderer.draw_reveal_screen(self)
#         else:
#             self.renderer.draw_game_state(self)
        
#         pygame.display.flip()

# # --- 4. PUNTO DE ENTRADA ---

# def main():
#     # Comprobar si existe el modelo antes de empezar
#     if not os.path.exists(MODEL_PATH):
#         print(f"Error: No se encontró el modelo en {MODEL_PATH}")
#         print("Por favor, asegúrate de que la variable MODEL_PATH en el script")
#         print("apunta a tu archivo .zip de modelo PPO entrenado.")
#         return
        
#     game = Game()
#     game.run()

# if __name__ == "__main__":
#     main()

# import os
# import sys
# import pygame
# import time
# from enum import Enum
# from stable_baselines3 import PPO
# import random
# import numpy as np

# # Importamos el entorno y las clases.
# try:
#     from mus_gym.mus_env import MusEnvironment
#     from mus_gym.mus_enums import MusAction, MusLance, LanceState
#     from mus_gym.mus_entities import Carta, Mano
#     from mus_gym.mus_config import PALOS, RANGOS_REALES
# except ImportError:
#     print("Error: No se pudo encontrar el paquete 'mus_gym'.")
#     sys.exit(1)

# # --- 1. CONFIGURACIÓN Y CONSTANTES ---
# pygame.init()
# pygame.font.init()

# # Colores
# COLOR_BLANCO = (255, 255, 255)
# COLOR_NEGRO = (0, 0, 0)
# COLOR_GRIS_CLARO = (200, 200, 200)
# COLOR_VERDE_FONDO = (0, 100, 25)
# COLOR_PANEL_FONDO = (30, 60, 30)
# COLOR_BOTON = (50, 50, 150)
# COLOR_BOTON_HOVER = (80, 80, 180)
# COLOR_BOTON_DESHABILITADO = (100, 100, 100)
# COLOR_AMARILLO = (255, 255, 0)
# COLOR_ROJO_CLARO = (255, 150, 150)
# COLOR_VERDE_CLARO = (150, 255, 150)
# COLOR_LOG_IA = (255, 180, 180)
# COLOR_LOG_HUMANO = (180, 255, 180)
# COLOR_LOG_SISTEMA = (200, 200, 255)

# # Pantalla
# ANCHO_JUEGO, ALTO_JUEGO = 900, 750 # Un poco más grande para mejor visión
# ANCHO_PANEL = 350
# ANCHO_TOTAL = ANCHO_JUEGO + ANCHO_PANEL
# ALTO_TOTAL = ALTO_JUEGO

# # Fuentes
# try:
#     FONT_GRANDE = pygame.font.SysFont('Arial', 32, bold=True)
#     FONT_MEDIANA = pygame.font.SysFont('Arial', 24)
#     FONT_PEQUENA = pygame.font.SysFont('Arial', 18)
#     FONT_LOG = pygame.font.SysFont('Consolas', 15)
# except pygame.error:
#     FONT_GRANDE = pygame.font.Font(None, 40)
#     FONT_MEDIANA = pygame.font.Font(None, 30)
#     FONT_PEQUENA = pygame.font.Font(None, 24)
#     FONT_LOG = pygame.font.Font(None, 20)

# # Ruta del modelo
# MODEL_PATH = "models/sheriff_colab/FINAL_MUS_MODEL.zip" # <-- AJUSTAR RUTA
# ASSET_PATH = os.path.join("assets", "cartas")
# CARD_SIZE = (85, 130)

# class GameState(Enum):
#     LOADING = 0
#     HUMAN_TURN = 1
#     AI_THINKING = 2
#     AI_STEP = 3
#     REVEAL_HAND = 4
#     GAME_OVER = 5

# class MusRenderer:
#     def __init__(self, pantalla):
#         self.pantalla = pantalla
#         self.card_images = {}
#         self.dorso_img = None
#         self.button_rects = {}
#         self.load_assets()

#     def load_assets(self):
#         print("Cargando imágenes de cartas...")
#         try:
#             for palo in PALOS:
#                 for rango in RANGOS_REALES:
#                     key = f"{rango}_{palo}"
#                     filename = f"{key}.png".lower()
#                     path = os.path.join(ASSET_PATH, filename)
#                     image = pygame.image.load(path).convert_alpha()
#                     self.card_images[key] = pygame.transform.scale(image, CARD_SIZE)
            
#             path = os.path.join(ASSET_PATH, "dorso.png")
#             self.dorso_img = pygame.image.load(path).convert_alpha()
#             self.dorso_img = pygame.transform.scale(self.dorso_img, CARD_SIZE)
#         except Exception as e:
#             print(f"--- ¡ERROR DE ASSETS! --- {e}")
#             pygame.quit()
#             sys.exit()

#     def draw_card(self, carta_obj, x, y, face_down=False):
#         if face_down or not carta_obj:
#             self.pantalla.blit(self.dorso_img, (x, y))
#         else:
#             key = f"{carta_obj.rango}_{carta_obj.palo}"
#             if key in self.card_images:
#                 self.pantalla.blit(self.card_images[key], (x, y))
#             else:
#                 pygame.draw.rect(self.pantalla, (255, 0, 0), (x, y, CARD_SIZE[0], CARD_SIZE[1]))

#     def draw_text(self, texto, font, color, x, y, center=False):
#         text_surface = font.render(texto, True, color)
#         text_rect = text_surface.get_rect()
#         if center:
#             text_rect.center = (x, y)
#         else:
#             text_rect.topleft = (x, y)
#         self.pantalla.blit(text_surface, text_rect)

#     def draw_game_state(self, game):
#         self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_JUEGO, ALTO_TOTAL))
#         self.pantalla.fill(COLOR_PANEL_FONDO, (ANCHO_JUEGO, 0, ANCHO_PANEL, ALTO_TOTAL))
#         pygame.draw.line(self.pantalla, COLOR_BLANCO, (ANCHO_JUEGO, 0), (ANCHO_JUEGO, ALTO_TOTAL), 2)

#         # Manos
#         mano_humano = game.env.manos[0]
#         # La mano de la IA está oculta
#         for i in range(4):
#             self.draw_card(None, 250 + i * 110, 30, face_down=True)
        
#         if mano_humano:
#             for i, carta in enumerate(mano_humano.cartas):
#                 self.draw_card(carta, 250 + i * 110, ALTO_JUEGO - 160)
        
#         # Info central
#         self.draw_turn_indicator(game)
#         self.draw_action_buttons(game)
#         self.draw_info_panel(game)

#     def draw_turn_indicator(self, game):
#         if game.state == GameState.HUMAN_TURN:
#             txt, col = "¡TU TURNO!", COLOR_AMARILLO
#         elif game.state == GameState.AI_THINKING:
#             txt, col = "IA PENSANDO...", COLOR_GRIS_CLARO
#         else:
#             txt, col = "...", COLOR_GRIS_CLARO

#         self.draw_text(txt, FONT_GRANDE, col, ANCHO_JUEGO // 2, 280, center=True)
        
#         lance_name = game.env.current_lance.name if game.env.current_lance else 'N/A'
#         self.draw_text(f"LANCE ACTUAL: {lance_name}", FONT_GRANDE, COLOR_BLANCO, ANCHO_JUEGO // 2, 330, center=True)
        
#         # Detalle apuesta
#         if game.env.lance_bet > 0:
#             bet_txt = f"Hay apuesta: {game.env.lance_bet} piedras"
#             if game.env.lance_bet >= game.env.target_score:
#                 bet_txt = "¡¡¡ ÓRDAGO !!!"
#             self.draw_text(bet_txt, FONT_GRANDE, COLOR_ROJO_CLARO, ANCHO_JUEGO // 2, 380, center=True)
        
#         # Quién es mano
#         es_mano = "Eres Mano" if game.env.mano_player_idx == 0 else "IA es Mano"
#         self.draw_text(f"({es_mano})", FONT_PEQUENA, COLOR_GRIS_CLARO, ANCHO_JUEGO // 2, 240, center=True)

#     def draw_action_buttons(self, game):
#         self.button_rects.clear()
#         es_turno_humano = (game.state == GameState.HUMAN_TURN)
#         legal_actions = game.env._get_legal_actions() if es_turno_humano else []
        
#         ancho, alto = 140, 50
#         start_x = (ANCHO_JUEGO - (len(legal_actions) * (ancho + 10))) // 2
#         mouse = pygame.mouse.get_pos()
        
#         for i, action in enumerate(legal_actions):
#             rect = pygame.Rect(start_x + i * (ancho + 10), 450, ancho, alto)
#             self.button_rects[action] = rect
            
#             color = COLOR_BOTON_HOVER if (es_turno_humano and rect.collidepoint(mouse)) else COLOR_BOTON
#             if not es_turno_humano: color = COLOR_BOTON_DESHABILITADO
            
#             pygame.draw.rect(self.pantalla, color, rect, border_radius=8)
#             pygame.draw.rect(self.pantalla, COLOR_BLANCO, rect, width=2, border_radius=8)
#             self.draw_text(action.name, FONT_MEDIANA, COLOR_BLANCO, rect.centerx, rect.centery, center=True)

#     def draw_info_panel(self, game):
#         x = ANCHO_JUEGO + 20
#         # Puntuación
#         self.draw_text(f"TU: {game.env.scores[0]}", FONT_GRANDE, COLOR_VERDE_CLARO, x, 30)
#         self.draw_text(f"IA: {game.env.scores[1]}", FONT_GRANDE, COLOR_ROJO_CLARO, x + 180, 30)
#         pygame.draw.line(self.pantalla, COLOR_GRIS_CLARO, (x, 70), (ANCHO_TOTAL-20, 70), 1)

#         # Info Mano Propia
#         m = game.env.manos[0]
#         if m:
#             self.draw_text("TU MANO:", FONT_MEDIANA, COLOR_AMARILLO, x, 90)
#             pares = m.get_pares_info()
#             juego = m.get_juego_valor()
#             self.draw_text(f"- {pares['tipo']}", FONT_PEQUENA, COLOR_BLANCO, x+10, 120)
#             self.draw_text(f"- Juego: {juego} ({'Sí' if m.tiene_juego() else 'No'})", FONT_PEQUENA, COLOR_BLANCO, x+10, 145)

#         # LOG
#         self.draw_text("HISTORIAL:", FONT_MEDIANA, COLOR_AMARILLO, x, 250)
#         pygame.draw.rect(self.pantalla, COLOR_NEGRO, (x, 280, ANCHO_PANEL-30, ALTO_TOTAL-300))
        
#         y_log = 290
#         for msg, col in game.message_log[-20:]: # Muestra últimos 20
#             self.draw_text(msg, FONT_LOG, col, x+5, y_log)
#             y_log += 18

#     def draw_game_over_screen(self, game):
#         # Reutilizamos la pantalla de revelar mano, pero con texto de GANADOR
#         self.pantalla.fill(COLOR_PANEL_FONDO, (0, 0, ANCHO_TOTAL, ALTO_TOTAL))
        
#         ganador = "¡¡ HAS GANADO LA PARTIDA !!" if game.env.scores[0] >= 30 else "LA IA HA GANADO LA PARTIDA"
#         color = COLOR_VERDE_CLARO if game.env.scores[0] >= 30 else COLOR_ROJO_CLARO
        
#         self.draw_text(ganador, FONT_GRANDE, color, ANCHO_TOTAL//2, 50, center=True)
#         self.draw_text(f"Puntuación Final: {game.env.scores[0]} vs {game.env.scores[1]}", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL//2, 100, center=True)

#         # Mostrar las cartas finales
#         self.draw_text("Manos Finales:", FONT_MEDIANA, COLOR_AMARILLO, ANCHO_TOTAL//2, 160, center=True)
        
#         # IA
#         mano_ia = game.env.manos[1] # Usamos la actual
#         if mano_ia:
#             for i, c in enumerate(mano_ia.cartas):
#                 self.draw_card(c, 250 + i * 110, 200, face_down=False)
#             info = f"IA: {mano_ia.get_pares_info()['tipo']} | Juego: {mano_ia.get_juego_valor()}"
#             self.draw_text(info, FONT_PEQUENA, COLOR_GRIS_CLARO, ANCHO_TOTAL//2, 340, center=True)

#         # Humano
#         mano_h = game.env.manos[0]
#         if mano_h:
#             for i, c in enumerate(mano_h.cartas):
#                 self.draw_card(c, 250 + i * 110, 400, face_down=False)
#             info = f"Tú: {mano_h.get_pares_info()['tipo']} | Juego: {mano_h.get_juego_valor()}"
#             self.draw_text(info, FONT_PEQUENA, COLOR_GRIS_CLARO, ANCHO_TOTAL//2, 540, center=True)

#         self.draw_text("Pulsa [ESPACIO] para salir", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL//2, 650, center=True)
#         pygame.display.flip()

#     def draw_reveal_screen(self, game):
#         # Similar a game over pero para fin de mano
#         self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_TOTAL, ALTO_TOTAL))
#         self.draw_text("FIN DE LA MANO - RESULTADOS", FONT_GRANDE, COLOR_AMARILLO, ANCHO_TOTAL//2, 40, center=True)
        
#         # IA
#         mano_ia = game.old_manos[1]
#         if mano_ia:
#             self.draw_text("IA tenía:", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL//2, 100, center=True)
#             for i, c in enumerate(mano_ia.cartas):
#                 self.draw_card(c, 250 + i * 110, 140, face_down=False)
#             p = mano_ia.get_pares_info()
#             self.draw_text(f"{p['tipo']} ({p['valor_principal']}) | Juego: {mano_ia.get_juego_valor()}", FONT_PEQUENA, COLOR_GRIS_CLARO, ANCHO_TOTAL//2, 280, center=True)

#         # Tu
#         mano_h = game.old_manos[0]
#         if mano_h:
#             self.draw_text("Tú tenías:", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL//2, 350, center=True)
#             for i, c in enumerate(mano_h.cartas):
#                 self.draw_card(c, 250 + i * 110, 390, face_down=False)
        
#         self.draw_text("Pulsa [ESPACIO] para siguiente mano...", FONT_MEDIANA, COLOR_AMARILLO, ANCHO_TOTAL//2, 600, center=True)
#         pygame.display.flip()


# class Game:
#     def __init__(self):
#         self.pantalla = pygame.display.set_mode((ANCHO_TOTAL, ALTO_TOTAL))
#         pygame.display.set_caption("Mus Master AI - Debug Mode")
#         self.clock = pygame.time.Clock()
#         self.renderer = MusRenderer(self.pantalla)
        
#         self.env = None
#         self.model = None
#         self.running = True
#         self.state = GameState.LOADING
        
#         self._last_lstm_states = None
#         self._last_episode_starts = np.ones((1,), dtype=bool)
        
#         self.message_log = []
#         self.old_manos = (None, None)
#         self.load_model_and_env()

#     def load_model_and_env(self):
#         if not os.path.exists(MODEL_PATH):
#             print("ERROR: Modelo no encontrado.")
#             return
#         from sb3_contrib import RecurrentPPO
#         self.model = RecurrentPPO.load(MODEL_PATH)
#         self.env = MusEnvironment(target_score=30)
#         self.add_log("Sistema: Modelo y Entorno cargados.", COLOR_LOG_SISTEMA)

#     def add_log(self, message, color):
#         timestamp = time.strftime("%H:%M:%S")
#         self.message_log.append((f"[{timestamp}] {message}", color))
#         print(f"LOG: {message}")

#     def run(self):
#         if not self.env: return
#         self.obs_vector, _ = self.env.reset()
#         self.state = self.get_state_from_env()
        
#         while self.running:
#             action = self.handle_events()
            
#             if self.state == GameState.HUMAN_TURN and action:
#                 self.step(action)
#             elif self.state == GameState.AI_THINKING:
#                 # Pequeño delay artificial para ver que piensa
#                 pygame.time.wait(600) 
#                 self.ai_step()
#             elif self.state == GameState.REVEAL_HAND and action == "CONTINUE":
#                  self.add_log("--- Nueva Mano ---", COLOR_LOG_SISTEMA)
#                  self.state = self.get_state_from_env()
#             elif self.state == GameState.GAME_OVER and action == "CONTINUE":
#                 self.running = False

#             self.renderer.draw_game_state(self)
#             if self.state == GameState.GAME_OVER:
#                 self.renderer.draw_game_over_screen(self)
#             elif self.state == GameState.REVEAL_HAND:
#                 self.renderer.draw_reveal_screen(self)
            
#             pygame.display.flip()
#             self.clock.tick(30)
#         pygame.quit()

#     def handle_events(self):
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT: self.running = False; return None
#             if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: return "CONTINUE"
#             if event.type == pygame.MOUSEBUTTONDOWN and self.state == GameState.HUMAN_TURN:
#                 for act, rect in self.renderer.button_rects.items():
#                     if rect.collidepoint(event.pos): return act
#             if (self.state in [GameState.GAME_OVER, GameState.REVEAL_HAND]) and \
#                (event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE)):
#                    return "CONTINUE"
#         return None

#     def step(self, action: MusAction):
#         # Guardar estado previo para logs de puntos
#         score_pre = self.env.scores.copy()
#         lance_pre = self.env.current_lance
        
#         # Log de acción
#         actor = "TU" if self.env.current_player_idx == 0 else "IA"
#         col = COLOR_LOG_HUMANO if actor == "TU" else COLOR_LOG_IA
        
#         # Mensajes más explicativos
#         txt_accion = action.name
#         if action == MusAction.ENVIDO: txt_accion = "Envida 2 piedras"
#         if action == MusAction.ORDAGO: txt_accion = "¡¡ ÓRDAGO !!"
        
#         self.add_log(f"{actor}: {txt_accion}", col)

#         # --- STEP ---
#         self.old_manos = (self.env.manos[0], self.env.manos[1]) # Guardar para Reveal
#         self.obs_vector, _, term, trunc, _ = self.env.step(action.value)
#         self._last_episode_starts[0] = term or trunc

#         # Logs de puntos ganados en este paso
#         if self.env.scores[0] > score_pre[0]:
#             diff = self.env.scores[0] - score_pre[0]
#             self.add_log(f"-> Ganas {diff} piedra(s)", COLOR_VERDE_CLARO)
#         if self.env.scores[1] > score_pre[1]:
#             diff = self.env.scores[1] - score_pre[1]
#             self.add_log(f"-> IA gana {diff} piedra(s)", COLOR_ROJO_CLARO)

#         # Cambio de lance
#         if self.env.current_lance != lance_pre and not (term or trunc):
#              self.add_log(f"Cambio a: {self.env.current_lance.name}", COLOR_LOG_SISTEMA)

#         # Estado siguiente
#         if term or trunc:
#             self.state = GameState.GAME_OVER
#             self.add_log("PARTIDA FINALIZADA", COLOR_AMARILLO)
#         elif self.env.current_lance == MusLance.GRANDE and lance_pre in [MusLance.PARES, MusLance.JUEGO]:
#              self.state = GameState.REVEAL_HAND
#         else:
#             self.state = self.get_state_from_env()

#     def ai_step(self):
#         obs_ia = self.env._get_observation(1)
#         act_val, self._last_lstm_states = self.model.predict(
#             obs_ia, state=self._last_lstm_states, episode_start=self._last_episode_starts, deterministic=False
#         )
#         # Validación
#         legal = self.env._get_legal_actions()
#         act = MusAction(act_val.item())
#         if act not in legal: act = random.choice(legal)
        
#         self.step(act)

#     def get_state_from_env(self):
#         return GameState.HUMAN_TURN if self.env.current_player_idx == 0 else GameState.AI_THINKING

# if __name__ == "__main__":
#     Game().run()
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

MODEL_PATH = "models/sheriff_colab/FINAL_MUS_MODEL.zip"
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

        self.draw_text(f"PUNTUACIÓN ACTUAL: TU {game.env.scores[0]} - IA {game.env.scores[1]}", FONT_GRANDE, COLOR_BLANCO, ANCHO_TOTAL//2, 550, center=True)

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
                obs, state=self.lstm_states, episode_start=self.episode_starts
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