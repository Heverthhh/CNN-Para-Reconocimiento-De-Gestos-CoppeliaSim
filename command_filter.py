"""command_filter.py - Filtro de seguridad por ventana, quórum y máquina de estados FSM."""

from collections import deque
from enum import Enum
from typing import Any, Dict, Optional


class FilterState(str, Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    WAIT_ZERO = "WAIT_ZERO"


class CommandFilter:
    """Máquina de estados finita que impide disparos múltiples y exige enclavamiento."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.umbral: float = float(cfg.get("confidence_threshold", 0.85))
        self.ventana_tam: int = int(cfg.get("window_size", 5))
        self.min_votos: int = int(cfg.get("min_votos", cfg.get("min_votes", 3)))
        self.cooldown_s: float = float(cfg.get("cooldown_s", cfg.get("cooldown_seconds", 1.2)))

        self.ventana: deque = deque(maxlen=self.ventana_tam)
        self.estado: FilterState = FilterState.IDLE
        self.last_cmd_time: float = 0.0
        self.robot_is_busy: bool = False

    def reset_to_idle(self) -> None:
        self.estado = FilterState.IDLE
        self.ventana.clear()

    def set_robot_busy(self, busy: bool) -> None:
        self.robot_is_busy = busy

    def update(
        self,
        clase: int,
        confianza: float,
        t: float,
        hand_present: bool = True,
    ) -> Dict[str, Any]:
        """Evalúa una predicción dentro de la máquina de estados."""
        # Solo forzar 0 si realmente no hay mano en el ROI
        c_efectiva = clase if hand_present else 0

        # Parada explícita por clase 0
        if c_efectiva == 0:
            self.ventana.append(0)
            if self.ventana.count(0) >= self.min_votos:
                self.estado = FilterState.IDLE
                return {
                    "comando_aceptado": 0,
                    "estado": self.estado.value,
                    "motivo_rechazo": None,
                }

        # 1. Estado BUSY
        if self.estado == FilterState.BUSY:
            if self.robot_is_busy:
                return {"comando_aceptado": None, "estado": self.estado.value, "motivo_rechazo": "robot ocupado"}
            if (t - self.last_cmd_time) < self.cooldown_s:
                return {"comando_aceptado": None, "estado": self.estado.value, "motivo_rechazo": "cooldown"}
            self.estado = FilterState.WAIT_ZERO
            self.ventana.clear()

        # 2. Estado WAIT_ZERO
        if self.estado == FilterState.WAIT_ZERO:
            self.ventana.append(c_efectiva)
            if self.ventana.count(0) >= self.min_votos or not hand_present:
                self.estado = FilterState.IDLE
                self.ventana.clear()
                return {"comando_aceptado": None, "estado": self.estado.value, "motivo_rechazo": None}
            return {"comando_aceptado": None, "estado": self.estado.value, "motivo_rechazo": "esperando clase 0"}

        # 3. Estado IDLE
        if self.robot_is_busy:
            return {"comando_aceptado": None, "estado": self.estado.value, "motivo_rechazo": "robot ocupado"}

        if confianza < self.umbral:
            self.ventana.clear()
            return {"comando_aceptado": None, "estado": self.estado.value, "motivo_rechazo": "baja confianza"}

        self.ventana.append(c_efectiva)

        if self.ventana.count(c_efectiva) < self.min_votos or len(self.ventana) < self.ventana_tam:
            return {"comando_aceptado": None, "estado": self.estado.value, "motivo_rechazo": "inestable"}

        # Disparo formal del comando activo
        self.last_cmd_time = t
        self.estado = FilterState.BUSY
        self.ventana.clear()

        return {
            "comando_aceptado": c_efectiva,
            "estado": self.estado.value,
            "motivo_rechazo": None,
        }