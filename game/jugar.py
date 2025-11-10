import os
import sys
import pygame
import random
import time  # Necesario para las pausas
from stable_baselines3 import PPO

# Importamos nuestro entorno y acciones
from entorno_mus import MusEnvironment, MusAction, Carta, PALOS, RANGOS_REALES

# --- 1. CONFIGURACIÓN DE PYGAME (Sin cambios) ---
pygame.init()
COLOR_BLANCO = (255, 255, 255)
COLOR_NEGRO = (0, 0, 0)
COLOR_GRIS_CLARO = (200, 200, 200)
COLOR_VERDE_FONDO = (0, 100, 25)
COLOR_BOTON = (50, 50, 150)
COLOR_BOTON_HOVER = (80, 80, 180)
COLOR_AMARILLO = (255, 255, 0)
COLOR_ROJO_CLARO = (255, 150, 150)

ANCHO, ALTO = 1000, 700
PANTALLA = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Mus: Humano vs IA")

try:
    FONT_GRANDE = pygame.font.SysFont('Arial', 32)
    FONT_MEDIANA = pygame.font.SysFont('Arial', 24)
    FONT_PEQUENA = pygame.font.SysFont('Arial', 18)
except:
    FONT_GRANDE = pygame.font.Font(None, 40)
    FONT_MEDIANA = pygame.font.Font(None, 30)
    FONT_PEQUENA = pygame.font.Font(None, 24)

# --- 2. CLASE MusRenderer (Modificada para mostrar cartas) ---

class MusRenderer:
    def __init__(self, pantalla):
        self.pantalla = pantalla
        self.button_rects = {} 
        self.card_images = {}
        self.card_size = (80, 120) 
        self.asset_path = os.path.join("assets", "cartas")

        print("Cargando imágenes de cartas...")
        try:
            for palo in PALOS:
                for rango in RANGOS_REALES:
                    key = f"{rango}_{palo}"
                    filename = f"{key}.png".lower() 
                    path = os.path.join(self.asset_path, filename)
                    image = pygame.image.load(path).convert_alpha()
                    image = pygame.transform.scale(image, self.card_size)
                    self.card_images[key] = image
            
            path = os.path.join(self.asset_path, "dorso.png")
            self.dorso_img = pygame.image.load(path).convert_alpha()
            self.dorso_img = pygame.transform.scale(self.dorso_img, self.card_size)
            print("¡Imágenes cargadas con éxito!")
            
        except FileNotFoundError as e:
            print(f"--- ¡ERROR DE ASSETS! ---")
            print(f"No se pudo encontrar la imagen: {e.filename}")
            pygame.quit()
            sys.exit()

    def draw_card(self, carta_obj, x, y, face_down=False):
        if face_down:
            self.pantalla.blit(self.dorso_img, (x, y))
        else:
            key = f"{carta_obj.rango}_{carta_obj.palo}"
            if key in self.card_images:
                self.pantalla.blit(self.card_images[key], (x, y))
            else:
                pygame.draw.rect(self.pantalla, (255, 0, 0), (x, y, self.card_size[0], self.card_size[1]))

    # --- ¡NUEVA FUNCIÓN! ---
    def reveal_hands(self, mano_humano, mano_ia):
        """Dibuja ambas manos boca arriba al final del lance."""
        print("Mostrando cartas al final de la mano...")
        self.pantalla.fill(COLOR_VERDE_FONDO)
        # Mano Humano (ya visible)
        for i, carta in enumerate(mano_humano.cartas):
            self.draw_card(carta, 300 + i * 100, ALTO - 150)
            
        # Mano IA (ahora boca arriba)
        for i, carta in enumerate(mano_ia.cartas):
            self.draw_card(carta, 300 + i * 100, 30, face_down=False) # <- face_down=False

        # Texto
        txt_reveal = FONT_GRANDE.render("FIN DE LA MANO", True, COLOR_AMARILLO)
        self.pantalla.blit(txt_reveal, (ANCHO // 2 - txt_reveal.get_width() // 2, 350))
        txt_continue = FONT_MEDIANA.render("Pulsa Enter para la siguiente mano...", True, COLOR_BLANCO)
        self.pantalla.blit(txt_continue, (ANCHO // 2 - txt_continue.get_width() // 2, 400))
        
        pygame.display.flip()

        # Esperar a que el humano pulse Enter
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    waiting = False
            pygame.time.wait(50) # No consumir 100% CPU

    # --- FUNCIÓN draw_game_state (Sin cambios) ---
    def draw_game_state(self, env, legal_actions, es_turno_humano, last_ia_action=None):
        self.pantalla.fill(COLOR_VERDE_FONDO)
        self.button_rects.clear() 

        # 1. Dibujar Manos
        if env.manos[0]:
            for i, carta in enumerate(env.manos[0].cartas):
                self.draw_card(carta, 300 + i * 100, ALTO - 150)
        
        if env.manos[1]:
            for i in range(4):
                # La mano de la IA siempre se dibuja boca abajo durante el juego
                self.draw_card(None, 300 + i * 100, 30, face_down=True) 

        # 2. Dibujar Puntuaciones y Estado
        txt_score = FONT_GRANDE.render(f"PUNTUACIÓN   [Humano: {env.scores[0]} | IA: {env.scores[1]}]", True, COLOR_BLANCO)
        self.pantalla.blit(txt_score, (ANCHO // 2 - txt_score.get_width() // 2, 200))
        txt_lance = FONT_MEDIANA.render(f"Lance: {env.current_lance} | Estado: {env.lance_state}", True, COLOR_BLANCO)
        self.pantalla.blit(txt_lance, (ANCHO // 2 - txt_lance.get_width() // 2, 250))
        txt_mano = FONT_MEDIANA.render(f"J{env.mano_player_idx} es 'mano'", True, COLOR_BLANCO)
        self.pantalla.blit(txt_mano, (ANCHO // 2 - txt_mano.get_width() // 2, 290))

        # 3. Dibujar Botones de Acción
        if es_turno_humano:
            txt_turno = FONT_GRANDE.render("¡ES TU TURNO!", True, COLOR_AMARILLO) 
            self.pantalla.blit(txt_turno, (ANCHO // 2 - txt_turno.get_width() // 2, 350))
            
            if env.lance_state == "BET_MADE" and last_ia_action:
                txt_IA_dijo = FONT_MEDIANA.render(f"La IA ha dicho: {last_ia_action.name}", True, COLOR_ROJO_CLARO)
                self.pantalla.blit(txt_IA_dijo, (ANCHO // 2 - txt_IA_dijo.get_width() // 2, 470))

            ancho_boton = 150
            alto_boton = 50
            num_botones = len(legal_actions)
            start_x = (ANCHO - (num_botones * (ancho_boton + 20) - 20)) // 2
            mouse_pos = pygame.mouse.get_pos()
            
            for i, action in enumerate(legal_actions):
                button_rect = pygame.Rect(start_x + i * (ancho_boton + 20), 400, ancho_boton, alto_boton)
                self.button_rects[action] = button_rect
                color = COLOR_BOTON_HOVER if button_rect.collidepoint(mouse_pos) else COLOR_BOTON
                pygame.draw.rect(self.pantalla, color, button_rect, border_radius=10)
                txt_accion = FONT_MEDIANA.render(action.name, True, COLOR_BLANCO)
                txt_rect = txt_accion.get_rect(center=button_rect.center)
                self.pantalla.blit(txt_accion, txt_rect)
        else:
            txt_turno = FONT_GRANDE.render("Pensando IA...", True, COLOR_GRIS_CLARO)
            self.pantalla.blit(txt_turno, (ANCHO // 2 - txt_turno.get_width() // 2, 350))

        pygame.display.flip()

    # --- FUNCIÓN wait_for_human_action (Sin cambios) ---
    def wait_for_human_action(self) -> MusAction:
        while True:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for action, rect in self.button_rects.items():
                        if rect.collidepoint(event.pos):
                            return action
            
            mouse_pos = pygame.mouse.get_pos()
            for action, button_rect in self.button_rects.items():
                color = COLOR_BOTON_HOVER if button_rect.collidepoint(mouse_pos) else COLOR_BOTON
                pygame.draw.rect(self.pantalla, color, button_rect, border_radius=10)
                txt_accion = FONT_MEDIANA.render(action.name, True, COLOR_BLANCO)
                txt_rect = txt_accion.get_rect(center=button_rect.center)
                self.pantalla.blit(txt_accion, txt_rect)
            pygame.display.flip()

# --- 3. EL BUCLE PRINCIPAL (Modificado) ---

# --- ¡NUEVO! Colores para la terminal ---
class TermColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def main():
    MODEL_PATH = "./models/mus_ppo_v3/mus_agent_v3_norm_1M.zip" # Carga el modelo que quieras probar
    
    if not os.path.exists(MODEL_PATH):
        print(f"{TermColors.FAIL}Error: No se encontró el modelo en {MODEL_PATH}{TermColors.ENDC}")
        return

    print(f"{TermColors.OKGREEN}Cargando modelo IA desde {MODEL_PATH}...{TermColors.ENDC}")
    model = PPO.load(MODEL_PATH)
    print(f"{TermColors.OKGREEN}Modelo cargado. ¡Empieza la partida!{TermColors.ENDC}")

    env = MusEnvironment(target_score=30)
    renderer = MusRenderer(PANTALLA)
    
    obs_vector, info = env.reset(seed=random.randint(0, 10000))
    done = False
    
    last_ia_action = None 
    last_lance = "GRANDE" # Para saber cuándo termina una mano
    old_manos = (None, None)

    while not done:
        current_player_id = env.current_player_idx
        es_turno_humano = (current_player_id == 0)
        legal_actions = env._get_legal_actions()
        
        # --- ¡MODIFICACIÓN CLAVE! ---
        # Guarda las manos ANTES de que el step() las pueda cambiar
        if env.manos[0]:
            old_manos = (env.manos[0], env.manos[1])
        
        # Comprueba el estado de la mano ANTERIOR
        if env.current_lance == "GRANDE" and last_lance == "JUEGO":
            # ¡Usa las manos guardadas (old_manos) para el reveal!
            renderer.reveal_hands(old_manos[0], old_manos[1])
        last_lance = env.current_lance

        renderer.draw_game_state(env, legal_actions, es_turno_humano, last_ia_action)

        action_value = None
        action_enum = None

        # --- Imprimir info clara en la terminal ---
# --- Imprimir info clara en la terminal (VERSIÓN DEPURACIÓN) ---
        print("\n" + "="*30)
        print(f"{TermColors.HEADER}PUNTUACIÓN: [Humano: {env.scores[0]} | IA: {env.scores[1]}]{TermColors.ENDC}")
        print(f"Turno de: {'Humano (J0)' if es_turno_humano else 'IA (J1)'}")
        print(f"Posición: J{env.mano_player_idx} es 'mano'")
        print(f"Lance: {TermColors.BOLD}{env.current_lance} | {env.lance_state}{TermColors.ENDC}")

        # --- ¡¡NUEVO BLOQUE DE DEPURACIÓN!! ---
        # Mostramos el estado de AMBAS manos, calculado AHORA MISMO.
        if env.manos[0]: # Asegurarse de que las manos están repartidas
            humano_mano = env.manos[0]
            ia_mano = env.manos[1]
            
            print(f"Mano Humano (J0): {humano_mano}")
            print(f"  -> Humano TIENE JUEGO?: {TermColors.BOLD}{humano_mano.tiene_juego()}{TermColors.ENDC} (Suma: {humano_mano.get_juego_valor()})")
            
            print(f"Mano IA (J1): {ia_mano}")
            print(f"  -> IA TIENE JUEGO?:     {TermColors.BOLD}{ia_mano.tiene_juego()}{TermColors.ENDC} (Suma: {ia_mano.get_juego_valor()})")
        # --- FIN DEL BLOQUE DE DEPURACIÓN ---
            
        print(f"{TermColors.OKGREEN}Acciones Legales para J{current_player_id}: {[a.name for a in legal_actions]}{TermColors.ENDC}")
        # --- Fin de la info en terminal ---
        # --- Fin de la info en terminal ---

        if es_turno_humano:
            action_enum = renderer.wait_for_human_action()
            action_value = action_enum.value
            last_ia_action = None 
            print(f"{TermColors.OKBLUE}TÚ ELIGES: {action_enum.name}{TermColors.ENDC}")
            
        else:
            obs_vector_ia = env._get_observation(player_idx=1) 
           # action_tuple = model.predict(obs_vector_ia, deterministic=True) # Juega su "mejor" jugada
            action_tuple = model.predict(obs_vector_ia, deterministic=False) # Descomenta para jugar contra la IA "farolera"
            
            action_value = action_tuple[0].item()
            action_enum = MusAction(action_value)
            last_ia_action = action_enum
            
            print(f"{TermColors.WARNING}LA IA ELIGE: {action_enum.name}{TermColors.ENDC}") # <-- ¡"Voz" de la IA!
            time.sleep(1.5) # Pausa de 1.5s para poder leer la jugada

        obs_vector, reward, terminated, truncated, info = env.step(action_value)
        done = terminated or truncated
        
        if reward != 0:
            print(f"{TermColors.OKGREEN}*** Recompensa para J{current_player_id}: {reward} ***{TermColors.ENDC}")

    # --- Fin de la partida (Sin cambios) ---
    print(f"{TermColors.HEADER}¡Partida terminada!{TermColors.ENDC}")
    # Mostrar las cartas de la última mano
    renderer.reveal_hands(env.manos[0], env.manos[1])
    
    while True:
        PANTALLA.fill(COLOR_VERDE_FONDO)
        texto_final = "¡HAS GANADO!" if env.scores[0] > env.scores[1] else "HA GANADO LA IA"
        txt_render = FONT_GRANDE.render(texto_final, True, COLOR_AMARILLO)
        txt_rect = txt_render.get_rect(center=(ANCHO // 2, ALTO // 2))
        PANTALLA.blit(txt_render, txt_rect)
        
        txt_salir = FONT_MEDIANA.render("Haz clic para salir", True, COLOR_BLANCO)
        txt_salir_rect = txt_salir.get_rect(center=(ANCHO // 2, ALTO // 2 + 50))
        PANTALLA.blit(txt_salir, txt_salir_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                pygame.quit()
                sys.exit()

if __name__ == "__main__":
    main()