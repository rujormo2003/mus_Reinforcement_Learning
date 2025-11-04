import os
import sys
import pygame
import random
from stable_baselines3 import PPO

# Importamos nuestro entorno y acciones
from entorno_mus import MusEnvironment, MusAction, Carta

# --- 1. CONFIGURACIÓN DE PYGAME ---
pygame.init()

# Colores
COLOR_BLANCO = (255, 255, 255)
COLOR_NEGRO = (0, 0, 0)
COLOR_GRIS_CLARO = (200, 200, 200)
COLOR_GRIS_OSCURO = (100, 100, 100)
COLOR_VERDE_FONDO = (0, 100, 25)
COLOR_BOTON = (50, 50, 150)
COLOR_BOTON_HOVER = (80, 80, 180)

# Pantalla
ANCHO, ALTO = 1000, 700
PANTALLA = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Mus: Humano vs IA")

# Fuentes
try:
    FONT_GRANDE = pygame.font.SysFont('Arial', 32)
    FONT_MEDIANA = pygame.font.SysFont('Arial', 24)
    FONT_PEQUENA = pygame.font.SysFont('Arial', 18)
except: # Fallback por si Arial no está
    FONT_GRANDE = pygame.font.Font(None, 40)
    FONT_MEDIANA = pygame.font.Font(None, 30)
    FONT_PEQUENA = pygame.font.Font(None, 24)

# --- 2. EL RENDERIZADOR (LA "VISTA") ---

class MusRenderer:
    def __init__(self, pantalla):
        self.pantalla = pantalla
        self.button_rects = {} # Almacena {MusAction: pygame.Rect}

    def draw_card(self, carta_obj, x, y, face_down=False):
        """Dibuja una sola carta (con texto, sin imágenes)."""
        ancho_carta, alto_carta = 80, 120
        card_rect = pygame.Rect(x, y, ancho_carta, alto_carta)
        
        if face_down:
            pygame.draw.rect(self.pantalla, COLOR_BOTON, card_rect, border_radius=5)
            pygame.draw.rect(self.pantalla, COLOR_BLANCO, card_rect, 3, 5)
        else:
            pygame.draw.rect(self.pantalla, COLOR_BLANCO, card_rect, border_radius=5)
            pygame.draw.rect(self.pantalla, COLOR_NEGRO, card_rect, 2, 5)
            
            # Dibujar texto de la carta
            txt_rango = FONT_PEQUENA.render(carta_obj.rango, True, COLOR_NEGRO)
            txt_palo = FONT_PEQUENA.render(carta_obj.palo, True, COLOR_NEGRO)
            self.pantalla.blit(txt_rango, (x + 10, y + 10))
            self.pantalla.blit(txt_palo, (x + 10, y + 40))

    def draw_game_state(self, env, legal_actions, es_turno_humano, last_ia_action=None): # <-- MODIFICADO
            """Dibuja toda la pantalla: manos, puntuaciones y botones."""
            self.pantalla.fill(COLOR_VERDE_FONDO)
            self.button_rects.clear() 

            # --- 1. Dibujar Manos (Sin cambios) ---
            if env.manos[0]:
                for i, carta in enumerate(env.manos[0].cartas):
                    self.draw_card(carta, 300 + i * 100, ALTO - 150)
            if env.manos[1]:
                for i in range(4):
                    self.draw_card(None, 300 + i * 100, 30, face_down=True)

            # --- 2. Dibujar Puntuaciones y Estado (Sin cambios) ---
            txt_score = FONT_GRANDE.render(f"PUNTUACIÓN   [Humano: {env.scores[0]} | IA: {env.scores[1]}]", True, COLOR_BLANCO)
            self.pantalla.blit(txt_score, (ANCHO // 2 - txt_score.get_width() // 2, 200))
            txt_lance = FONT_MEDIANA.render(f"Lance: {env.current_lance} | Estado: {env.lance_state}", True, COLOR_BLANCO)
            self.pantalla.blit(txt_lance, (ANCHO // 2 - txt_lance.get_width() // 2, 250))
            txt_mano = FONT_MEDIANA.render(f"J{env.mano_player_idx} es 'mano'", True, COLOR_BLANCO)
            self.pantalla.blit(txt_mano, (ANCHO // 2 - txt_mano.get_width() // 2, 290))

            # --- 3. Dibujar Botones de Acción (MODIFICADO) ---
            if es_turno_humano:
                txt_turno = FONT_GRANDE.render("¡ES TU TURNO!", True, (255, 255, 0)) # Amarillo
                self.pantalla.blit(txt_turno, (ANCHO // 2 - txt_turno.get_width() // 2, 350))
                
                # --- ¡¡AQUÍ ESTÁ EL ARREGLO!! ---
                # Si estamos respondiendo a una apuesta, mostrar qué dijo la IA
                if env.lance_state == "BET_MADE" and last_ia_action:
                    txt_IA_dijo = FONT_MEDIANA.render(f"La IA ha dicho: {last_ia_action.name}", True, (255, 150, 150)) # Rojo claro
                    self.pantalla.blit(txt_IA_dijo, (ANCHO // 2 - txt_IA_dijo.get_width() // 2, 470))
                # --- FIN DEL ARREGLO ---

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

    def wait_for_human_action(self) -> MusAction:
        """
        Bucle de espera. Pausa el juego hasta que el humano hace clic.
        Devuelve la MusAction que el humano ha elegido.
        """
        while True:
            # Volvemos a dibujar los botones para el efecto "hover"
            pygame.event.pump() # Procesar eventos para que la ventana no se cuelgue
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Comprobar si el clic fue en un botón
                    for action, rect in self.button_rects.items():
                        if rect.collidepoint(event.pos):
                            return action # ¡Acción seleccionada!
                            
            # Esto es necesario para que el hover se actualice
            # (No es lo más eficiente, pero es lo más claro)
            # Re-renderizamos los botones (pero no toda la pantalla)
            mouse_pos = pygame.mouse.get_pos()
            for action, button_rect in self.button_rects.items():
                color = COLOR_BOTON_HOVER if button_rect.collidepoint(mouse_pos) else COLOR_BOTON
                pygame.draw.rect(self.pantalla, color, button_rect, border_radius=10)
                txt_accion = FONT_MEDIANA.render(action.name, True, COLOR_BLANCO)
                txt_rect = txt_accion.get_rect(center=button_rect.center)
                self.pantalla.blit(txt_accion, txt_rect)
            pygame.display.flip()


# --- 3. EL BUCLE PRINCIPAL DEL JUEGO ---

def main():
    MODEL_PATH = "./models/mus_ppo/mus_agent_1M.zip" 
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    print(f"Cargando modelo IA desde {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)
    print("Modelo cargado. ¡Empieza la partida!")

    env = MusEnvironment(target_score=30)
    renderer = MusRenderer(PANTALLA)
    
    obs_vector, info = env.reset(seed=random.randint(0, 10000))
    done = False
    
    last_ia_action = None # <-- NUEVA LÍNEA: Para recordar qué hizo la IA
    
    while not done:
        current_player_id = env.current_player_idx
        es_turno_humano = (current_player_id == 0)
        legal_actions = env._get_legal_actions()
        
        # MODIFICADO: Pasamos la última acción de la IA al renderizador
        renderer.draw_game_state(env, legal_actions, es_turno_humano, last_ia_action)

        action_value = None
        action_enum = None # <-- NUEVA LÍNEA: Para almacenar la acción como Enum

        if es_turno_humano:
            action_enum = renderer.wait_for_human_action()
            action_value = action_enum.value
            last_ia_action = None # Limpiamos la acción de la IA, ya que hemos respondido
            
        else:
            obs_vector_ia = env._get_observation(player_idx=1) 
            action_tuple = model.predict(obs_vector_ia, deterministic=True)
            action_value = action_tuple[0].item()
            
            action_enum = MusAction(action_value) # <-- NUEVA LÍNEA
            last_ia_action = action_enum          # <-- NUEVA LÍNEA: Recordamos la acción
            
            pygame.time.wait(1000) 

        obs_vector, reward, terminated, truncated, info = env.step(action_value)
        done = terminated or truncated

    # --- Fin de la partida (Sin cambios) ---
    print("¡Partida terminada!")
    while True:
        texto_final = "¡HAS GANADO!" if env.scores[0] > env.scores[1] else "HA GANADO LA IA"
        txt_render = FONT_GRANDE.render(texto_final, True, (255, 255, 0))
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