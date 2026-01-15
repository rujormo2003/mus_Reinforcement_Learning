import gymnasium as gym
from sb3_contrib import RecurrentPPO
from .mus_curriculum_env import MusEnvCurriculum

class MusEnvSelfPlay(MusEnvCurriculum):
    """
    Entorno Mus-v3: Self-Play dinámico.
    Permite actualizar la política del oponente en tiempo de ejecución.
    """
    def __init__(self, target_score=30, opponent_path=None):
        if opponent_path is None:
            raise ValueError("MusEnvSelfPlay requiere 'opponent_path' inicial.")
            
        self.opponent_policy = self._load_policy(opponent_path)
        
        super().__init__(target_score=target_score, opponent_policy=self.opponent_policy)

    def _load_policy(self, path):
        """ Método auxiliar para cargar el modelo de forma segura """
        return RecurrentPPO.load(path)

    def update_opponent_model(self, new_model_path):
        """ 
        Esta función será llamada por el Callback cada X pasos.
        Actualiza el 'cerebro' del oponente.
        """
        self.oponente_actual = self._load_policy(new_model_path)