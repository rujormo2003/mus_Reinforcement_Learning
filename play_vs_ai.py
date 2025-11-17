import os
import sys
import pygame
import time
from enum import Enum
from stable_baselines3 import PPO
import random

# Importamos el entorno y las clases.
try:
    # (Asumiendo que este script está en el directorio raíz,
    # y 'mus_gym' es un subdirectorio)
    from mus_gym.mus_env import MusEnvironment
    from mus_gym.mus_enums import MusAction, MusLance, LanceState
    from mus_gym.mus_entities import Carta, Mano
    from mus_gym.mus_config import PALOS, RANGOS_REALES
except ImportError:
    print("Error: No se pudo encontrar el paquete 'mus_gym'.")
    print("Asegúrate de que el script está en el directorio correcto (fuera de 'mus_gym')")
    print("y que 'mus_gym' es un paquete (contiene __init__.py)")
    sys.exit(1)

# --- 1. CONFIGURACIÓN Y CONSTANTES ---
pygame.init()
pygame.font.init()

# Colores
COLOR_BLANCO = (255, 255, 255)
COLOR_NEGRO = (0, 0, 0)
COLOR_GRIS_CLARO = (200, 200, 200)
COLOR_VERDE_FONDO = (0, 100, 25)
COLOR_PANEL_FONDO = (30, 60, 30)
COLOR_BOTON = (50, 50, 150)
COLOR_BOTON_HOVER = (80, 80, 180)
COLOR_BOTON_DESHABILITADO = (100, 100, 100)
COLOR_AMARILLO = (255, 255, 0)
COLOR_ROJO_CLARO = (255, 150, 150)
COLOR_LOG_IA = (255, 180, 180)
COLOR_LOG_HUMANO = (180, 255, 180)
COLOR_LOG_SISTEMA = (200, 200, 255)

# Pantalla
ANCHO_JUEGO, ALTO_JUEGO = 800, 700
ANCHO_PANEL = 300
ANCHO_TOTAL = ANCHO_JUEGO + ANCHO_PANEL
ALTO_TOTAL = ALTO_JUEGO

# Fuentes
try:
    FONT_GRANDE = pygame.font.SysFont('Arial', 32)
    FONT_MEDIANA = pygame.font.SysFont('Arial', 24)
    FONT_PEQUENA = pygame.font.SysFont('Arial', 18)
    FONT_LOG = pygame.font.SysFont('Consolas', 16)
except pygame.error:
    FONT_GRANDE = pygame.font.Font(None, 40)
    FONT_MEDIANA = pygame.font.Font(None, 30)
    FONT_PEQUENA = pygame.font.Font(None, 24)
    FONT_LOG = pygame.font.Font(None, 22)

# Ruta del modelo
MODEL_PATH = "recurrent_ppo_mus_v2.zip" # <-- CAMBIA ESTO A LA RUTA DE TU MODELO
ASSET_PATH = os.path.join("assets", "cartas")
CARD_SIZE = (80, 120)

class GameState(Enum):
    """ Máquina de estados para controlar el flujo del juego """
    LOADING = 0
    HUMAN_TURN = 1
    AI_THINKING = 2
    AI_STEP = 3
    REVEAL_HAND = 4
    GAME_OVER = 5

# --- 2. CLASE MusRenderer ---

class MusRenderer:
    """ Gestiona toda la carga de assets y la lógica de dibujado. """
    def __init__(self, pantalla):
        self.pantalla = pantalla
        self.card_images = {}
        self.dorso_img = None
        self.button_rects = {}
        self.load_assets()

    def load_assets(self):
        print("Cargando imágenes de cartas...")
        try:
            for palo in PALOS:
                for rango in RANGOS_REALES:
                    key = f"{rango}_{palo}"
                    filename = f"{key}.png".lower()
                    path = os.path.join(ASSET_PATH, filename)
                    image = pygame.image.load(path).convert_alpha()
                    self.card_images[key] = pygame.transform.scale(image, CARD_SIZE)
            
            path = os.path.join(ASSET_PATH, "dorso.png")
            self.dorso_img = pygame.image.load(path).convert_alpha()
            self.dorso_img = pygame.transform.scale(self.dorso_img, CARD_SIZE)
            print("¡Imágenes cargadas con éxito!")
        except Exception as e:
            print(f"--- ¡ERROR DE ASSETS! ---")
            print(f"No se pudo cargar una imagen: {e}")
            print(f"Asegúrate de que la carpeta '{ASSET_PATH}' existe y contiene las imágenes.")
            pygame.quit()
            sys.exit()

    def draw_card(self, carta_obj, x, y, face_down=False):
        if face_down or not carta_obj:
            self.pantalla.blit(self.dorso_img, (x, y))
        else:
            key = f"{carta_obj.rango}_{carta_obj.palo}"
            if key in self.card_images:
                self.pantalla.blit(self.card_images[key], (x, y))
            else:
                pygame.draw.rect(self.pantalla, (255, 0, 0), (x, y, CARD_SIZE[0], CARD_SIZE[1]))

    def draw_text(self, texto, font, color, x, y, center=False):
        text_surface = font.render(texto, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.pantalla.blit(text_surface, text_rect)

    def draw_game_state(self, game):
        """ Dibuja el estado principal del juego (tablero y panel) """
        
        # 1. Fondo del tablero
        self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_JUEGO, ALTO_TOTAL))
        # 2. Fondo del panel
        self.pantalla.fill(COLOR_PANEL_FONDO, (ANCHO_JUEGO, 0, ANCHO_PANEL, ALTO_TOTAL))
        pygame.draw.line(self.pantalla, COLOR_BLANCO, (ANCHO_JUEGO, 0), (ANCHO_JUEGO, ALTO_TOTAL), 2)

        # 3. Dibujar Manos
        mano_humano = game.env.manos[0]
        mano_ia = game.env.manos[1]
        
        # Mano IA (Arriba)
        for i in range(4):
            self.draw_card(None, 200 + i * 100, 30, face_down=True)
        
        # Mano Humano (Abajo)
        if mano_humano:
            for i, carta in enumerate(mano_humano.cartas):
                self.draw_card(carta, 200 + i * 100, ALTO_JUEGO - 150)
        
        # 4. Dibujar Indicador de Turno y Estado
        self.draw_turn_indicator(game)

        # 5. Dibujar Botones de Acción
        self.draw_action_buttons(game)

        # 6. Dibujar Panel de Información
        self.draw_info_panel(game)

    def draw_turn_indicator(self, game):
        """ Dibuja el estado actual del juego en el centro """
        if game.state == GameState.HUMAN_TURN:
            texto_turno = "¡ES TU TURNO!"
            color_turno = COLOR_AMARILLO
        elif game.state == GameState.AI_THINKING:
            texto_turno = "IA PENSANDO..."
            color_turno = COLOR_GRIS_CLARO
        else:
            texto_turno = "..."
            color_turno = COLOR_GRIS_CLARO

        self.draw_text(texto_turno, FONT_GRANDE, color_turno, ANCHO_JUEGO // 2, 250, center=True)
        
        # Info del Lance
        lance_txt = f"Lance: {game.env.current_lance.name if game.env.current_lance else 'N/A'}"
        estado_txt = f"Estado: {game.env.current_lance_state.name}"
        apuesta_txt = f"Apuesta: {game.env.lance_bet} puntos"

        self.draw_text(lance_txt, FONT_MEDIANA, COLOR_BLANCO, ANCHO_JUEGO // 2, 300, center=True)
        self.draw_text(estado_txt, FONT_MEDIANA, COLOR_BLANCO, ANCHO_JUEGO // 2, 330, center=True)
        if game.env.lance_bet > 0:
            self.draw_text(apuesta_txt, FONT_MEDIANA, COLOR_AMARILLO, ANCHO_JUEGO // 2, 360, center=True)
        
        # Quién es "mano"
        mano_txt = f"JUGADOR {game.env.mano_player_idx} ES 'MANO'"
        self.draw_text(mano_txt, FONT_PEQUENA, COLOR_GRIS_CLARO, ANCHO_JUEGO // 2, 200, center=True)

    def draw_action_buttons(self, game):
        """ Dibuja los botones de acción para el humano """
        self.button_rects.clear()
        
        es_turno_humano = (game.state == GameState.HUMAN_TURN)
        legal_actions = game.env._get_legal_actions() if es_turno_humano else []
        
        ancho_boton = 150
        alto_boton = 50
        num_botones = len(legal_actions)
        start_x = (ANCHO_JUEGO - (num_botones * (ancho_boton + 20) - 20)) // 2
        mouse_pos = pygame.mouse.get_pos()
        
        for i, action in enumerate(legal_actions):
            button_rect = pygame.Rect(start_x + i * (ancho_boton + 20), 400, ancho_boton, alto_boton)
            self.button_rects[action] = button_rect
            
            color = COLOR_BOTON
            if es_turno_humano:
                if button_rect.collidepoint(mouse_pos):
                    color = COLOR_BOTON_HOVER
            else:
                color = COLOR_BOTON_DESHABILITADO
                
            pygame.draw.rect(self.pantalla, color, button_rect, border_radius=10)
            self.draw_text(action.name, FONT_MEDIANA, COLOR_BLANCO, button_rect.centerx, button_rect.centery, center=True)

    def draw_info_panel(self, game):
        """ Dibuja el panel lateral con puntuaciones, info de mano y log """
        x_base = ANCHO_JUEGO + 20
        
        # 1. Puntuaciones
        self.draw_text("PUNTUACIÓN", FONT_GRANDE, COLOR_AMARILLO, x_base, 20)
        self.draw_text(f"Humano: {game.env.scores[0]}", FONT_MEDIANA, COLOR_BLANCO, x_base + 10, 60)
        self.draw_text(f"IA: {game.env.scores[1]}", FONT_MEDIANA, COLOR_BLANCO, x_base + 10, 90)
        
        # 2. Info de Mano
        self.draw_text("INFO DE MANO", FONT_GRANDE, COLOR_AMARILLO, x_base, 150)
        
        mano_humano = game.env.manos[0]
        mano_ia = game.env.manos[1]
        
        # Info Humano
        self.draw_text("Humano:", FONT_MEDIANA, COLOR_BLANCO, x_base + 10, 190)
        if mano_humano:
            info_pares_h = mano_humano.get_pares_info()
            pares_txt_h = f"Pares: {info_pares_h['tipo']}"
            juego_txt_h = f"Juego: {'Sí' if mano_humano.tiene_juego() else 'No'} ({mano_humano.get_juego_valor()})"
            self.draw_text(pares_txt_h, FONT_PEQUENA, COLOR_GRIS_CLARO, x_base + 20, 220)
            self.draw_text(juego_txt_h, FONT_PEQUENA, COLOR_GRIS_CLARO, x_base + 20, 240)
        
        # Info IA (Oculta)
        self.draw_text("IA:", FONT_MEDIANA, COLOR_BLANCO, x_base + 10, 280)
        if mano_ia:
            self.draw_text("Pares: ???", FONT_PEQUENA, COLOR_GRIS_CLARO, x_base + 20, 310)
            self.draw_text("Juego: ???", FONT_PEQUENA, COLOR_GRIS_CLARO, x_base + 20, 330)
            
        # 3. Log de Acciones
        self.draw_text("LOG DE ACCIONES", FONT_GRANDE, COLOR_AMARILLO, x_base, 400)
        pygame.draw.rect(self.pantalla, COLOR_NEGRO, (x_base - 5, 435, ANCHO_PANEL - 30, ALTO_TOTAL - 450))
        
        y_log = 440
        max_logs = 12
        # Mostrar los últimos 'max_logs' mensajes
        for msg, color in game.message_log[-max_logs:]:
            self.draw_text(msg, FONT_LOG, color, x_base, y_log)
            y_log += FONT_LOG.get_height()

    def draw_reveal_screen(self, game):
        """ Dibuja la pantalla de fin de mano, mostrando toda la info """
        self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_TOTAL, ALTO_TOTAL))
        
        mano_humano = game.old_manos[0]
        mano_ia = game.old_manos[1]

        self.draw_text("FIN DE LA MANO", FONT_GRANDE, COLOR_AMARILLO, ANCHO_TOTAL // 2, 50, center=True)

        # Cartas IA
        self.draw_text("Mano de la IA:", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL // 2, 120, center=True)
        if mano_ia:
            for i, carta in enumerate(mano_ia.cartas):
                self.draw_card(carta, 200 + i * 100, 160, face_down=False)
            
            info_pares_ia = mano_ia.get_pares_info()
            pares_txt_ia = f"Pares: {info_pares_ia['tipo']} ({info_pares_ia['valor_principal']})"
            juego_txt_ia = f"Juego: {'Sí' if mano_ia.tiene_juego() else 'No'} ({mano_ia.get_juego_valor()})"
            self.draw_text(pares_txt_ia, FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, 300, center=True)
            self.draw_text(juego_txt_ia, FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, 330, center=True)

        # Cartas Humano
        self.draw_text("Tu Mano:", FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL // 2, 400, center=True)
        if mano_humano:
            for i, carta in enumerate(mano_humano.cartas):
                self.draw_card(carta, 200 + i * 100, 440, face_down=False)

            info_pares_h = mano_humano.get_pares_info()
            pares_txt_h = f"Pares: {info_pares_h['tipo']} ({info_pares_h['valor_principal']})"
            juego_txt_h = f"Juego: {'Sí' if mano_humano.tiene_juego() else 'No'} ({mano_humano.get_juego_valor()})"
            self.draw_text(pares_txt_h, FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, 580, center=True)
            self.draw_text(juego_txt_h, FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, 610, center=True)

        self.draw_text("Pulsa [ESPACIO] para continuar...", FONT_MEDIANA, COLOR_AMARILLO, ANCHO_TOTAL // 2, 670, center=True)
        pygame.display.flip()

    def draw_game_over_screen(self, game):
        """ Dibuja la pantalla de fin de partida """
        self.pantalla.fill(COLOR_VERDE_FONDO, (0, 0, ANCHO_TOTAL, ALTO_TOTAL))
        
        texto_final = "¡HAS GANADO!" if game.env.scores[0] > game.env.scores[1] else "HA GANADO LA IA"
        self.draw_text(texto_final, FONT_GRANDE, COLOR_AMARILLO, ANCHO_TOTAL // 2, ALTO_TOTAL // 2 - 50, center=True)
        
        score_txt = f"Puntuación Final: {game.env.scores[0]} - {game.env.scores[1]}"
        self.draw_text(score_txt, FONT_MEDIANA, COLOR_BLANCO, ANCHO_TOTAL // 2, ALTO_TOTAL // 2, center=True)
        
        self.draw_text("Haz clic o pulsa [ESPACIO] para salir", FONT_MEDIANA, COLOR_GRIS_CLARO, ANCHO_TOTAL // 2, ALTO_TOTAL // 2 + 50, center=True)
        pygame.display.flip()

# --- 3. CLASE Game (El motor) ---

class Game:
    """
    Gestiona el bucle principal del juego, el estado,
    el entorno de Gym y el modelo de SB3.
    """
    def __init__(self):
        self.pantalla = pygame.display.set_mode((ANCHO_TOTAL, ALTO_TOTAL))
        pygame.display.set_caption("Mus: Humano vs IA")
        self.clock = pygame.time.Clock()
        self.renderer = MusRenderer(self.pantalla)
        
        self.env = None
        self.model = None
        self.running = True
        self.state = GameState.LOADING
        
        self.obs_vector = None
        self.last_ia_action = None
        self.old_manos = (None, None)
        self.last_lance = None
        
        self.message_log = []
        self.ai_think_timer = 0
        
        self.load_model_and_env()
        self.add_log("¡Bienvenido a Mus vs IA!", COLOR_LOG_SISTEMA)

    def load_model_and_env(self):
        if not os.path.exists(MODEL_PATH):
            print(f"Error: No se encontró el modelo en {MODEL_PATH}")
            self.add_log(f"Error: Modelo no encontrado en {MODEL_PATH}", COLOR_ROJO_CLARO)
            self.running = False
            return
            
        try:
            print(f"Cargando modelo IA desde {MODEL_PATH}...")
            self.model = PPO.load(MODEL_PATH)
            print("Modelo cargado.")
            self.env = MusEnvironment(target_score=30)
            print("Entorno creado.")
        except Exception as e:
            print(f"Error al cargar modelo o entorno: {e}")
            self.add_log(f"Error al cargar: {e}", COLOR_ROJO_CLARO)
            self.running = False

    def add_log(self, message, color):
        """ Añade un mensaje al log en pantalla """
        print(f"LOG: {message}") # Mantenemos el log en terminal por si acaso
        self.message_log.append((message, color))
        if len(self.message_log) > 20: # Limitar el historial
            self.message_log.pop(0)

    def run(self):
        """ Bucle principal del juego """
        if not self.running:
            return # Fallo al cargar
            
        self.obs_vector, info = self.env.reset(seed=random.randint(0, 10000))
        self.state = self.get_state_from_env()
        self.last_lance = self.env.current_lance

        while self.running:
            # 1. Manejar Eventos (Input)
            action = self.handle_events()
            
            # 2. Actualizar Estado del Juego (Lógica)
            if self.state == GameState.HUMAN_TURN:
                if action:
                    self.execute_step(action)
            
            elif self.state == GameState.AI_THINKING:
                if pygame.time.get_ticks() > self.ai_think_timer:
                    self.execute_ai_step()
            
            elif self.state == GameState.REVEAL_HAND:
                if action == "CONTINUE":
                    self.state = self.get_state_from_env()
                    self.add_log("--- Nueva Mano ---", COLOR_LOG_SISTEMA)

            elif self.state == GameState.GAME_OVER:
                if action: # Cualquier acción (clic, espacio)
                    self.running = False
            
            # 3. Dibujar (Renderizado)
            self.render()
            
            # 4. Controlar FPS
            self.clock.tick(30) # 30 FPS
            
        pygame.quit()
        sys.exit()

    def handle_events(self):
        """ Procesa inputs de Pygame (clics, salir, etc.) """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return None
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                    return "CONTINUE" # Para pantallas de "Revelar" o "Game Over"

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.state == GameState.GAME_OVER:
                    return "CONTINUE"
                    
                if self.state == GameState.HUMAN_TURN:
                    mouse_pos = event.pos
                    for action, rect in self.renderer.button_rects.items():
                        if rect.collidepoint(mouse_pos):
                            return action # Devuelve la MusAction
        return None

    def execute_step(self, action: MusAction):
        """ 
        Ejecuta un paso en el entorno y actualiza nuestro estado del juego.
        Esta función es llamada tanto por el humano como por la IA.
        """
        
        # Guardar estado ANTES del paso
        self.old_manos = (self.env.manos[0], self.env.manos[1])
        self.last_lance = self.env.current_lance
        old_scores = self.env.scores.copy()
        
        # Identificar quién es
        es_humano = (self.env.current_player_idx == 0)
        actor_name = "Humano" if es_humano else "IA"
        actor_color = COLOR_LOG_HUMANO if es_humano else COLOR_LOG_IA
        
        self.add_log(f"{actor_name}: {action.name}", actor_color)
        
        # --- Ejecutar el paso en el entorno ---
        self.obs_vector, reward, terminated, truncated, info = self.env.step(action.value)
        # -------------------------------------
        
        self.last_ia_action = action if not es_humano else None

        # --- Comprobar cambios de puntuación ---
        # Esto es clave para informar al usuario de puntos "invisibles"
        if self.env.scores[0] > old_scores[0]:
            diff = self.env.scores[0] - old_scores[0]
            self.add_log(f"Humano gana {diff} punto(s)", COLOR_LOG_SISTEMA)
        if self.env.scores[1] > old_scores[1]:
            diff = self.env.scores[1] - old_scores[1]
            self.add_log(f"IA gana {diff} punto(s)", COLOR_LOG_SISTEMA)

        # --- Decidir el próximo estado ---
        if terminated or truncated:
            self.state = GameState.GAME_OVER
            self.add_log("¡¡¡PARTIDA TERMINADA!!!", COLOR_LOG_SISTEMA)
        
        # -----------------------------------------------------------------
        # --- INICIO DE LA CORRECCIÓN ---
        #
        # Si el lance actual es GRANDE, y el lance anterior
        # fue uno de los últimos lances (CHICA, PARES, o JUEGO),
        # significa que la mano ha terminado y hemos dado la vuelta.
        elif self.env.current_lance == MusLance.GRANDE and self.last_lance in [MusLance.CHICA, MusLance.PARES, MusLance.JUEGO]:
            self.state = GameState.REVEAL_HAND
            self.add_log("Fin de la mano. Mostrando cartas...", COLOR_LOG_SISTEMA)
        #
        # --- FIN DE LA CORRECCIÓN ---
        # -----------------------------------------------------------------
        
        else:
            self.state = self.get_state_from_env()

    def execute_ai_step(self):
        """ Pide una acción a la IA y la ejecuta """
        self.state = GameState.AI_STEP # Cambiar de "THINKING" a "STEP"
        
        # Usamos _get_observation del entorno base, que ya crea el vector
        obs_ia = self.env._get_observation(player_idx=1)
        
        # Juega de forma no-determinista (¡más divertido!)
        action_val, _ = self.model.predict(obs_ia, deterministic=False) 
        action_enum = MusAction(action_val.item())
        
        # Validar que la IA no haga trampas (por si acaso)
        legal_actions = self.env._get_legal_actions()
        if action_enum not in legal_actions:
            self.add_log(f"IA intentó acción ilegal: {action_enum.name}", COLOR_ROJO_CLARO)
            action_enum = random.choice(legal_actions) # Elegir una legal al azar
            
        self.execute_step(action_enum)

    def get_state_from_env(self):
        """ Decide el estado del juego basado en el turno del entorno """
        if self.env.current_player_idx == 0:
            return GameState.HUMAN_TURN
        else:
            # Iniciar el temporizador para que la IA "piense"
            self.ai_think_timer = pygame.time.get_ticks() + 1500 # 1.5 segundos
            return GameState.AI_THINKING

    def render(self):
        """ Dibuja la pantalla apropiada según el estado """
        
        if self.state == GameState.GAME_OVER:
            self.renderer.draw_game_over_screen(self)
        
        elif self.state == GameState.REVEAL_HAND:
            self.renderer.draw_reveal_screen(self)
        
        else:
            # Estados de juego normales (Human, AI_Thinking, AI_Step)
            self.renderer.draw_game_state(self)
        
        pygame.display.flip()

# --- 4. PUNTO DE ENTRADA ---

def main():
    # Comprobar si existe el modelo antes de empezar
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        print("Por favor, asegúrate de que la variable MODEL_PATH en el script")
        print("apunta a tu archivo .zip de modelo PPO entrenado.")
        return
        
    game = Game()
    game.run()

if __name__ == "__main__":
    main()