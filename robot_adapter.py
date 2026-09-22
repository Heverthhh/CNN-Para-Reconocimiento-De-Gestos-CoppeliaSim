"""robot_adapter.py - Adaptador desacoplado para CoppeliaSim vía API Remota ZMQ."""

from abc import ABC, abstractmethod
import math
import threading
import time
from typing import Any, Dict, List, Optional

import yaml

CONFIG_PATH = "config.yaml"


class RobotAdapter(ABC):
    """Clase abstracta base para la interfaz de control robótico."""

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> None:
        pass

    @abstractmethod
    def execute(self, comando: int) -> Dict[str, Any]:
        """Ejecuta un comando (0: Stop, 1..3: Articulación, 4: Pinza)."""
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        """True si el manipulador se encuentra en trayectoria de movimiento."""
        pass

    @abstractmethod
    def stop(self) -> Dict[str, Any]:
        """Cancela inmediatamente cualquier movimiento en curso."""
        pass

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        """Devuelve telemetría articular y estado actual."""
        pass


class MockAdapter(RobotAdapter):
    """Adaptador simulado para pruebas en seco sin conexión física ni simulación."""

    def __init__(self) -> None:
        self._busy = False
        self._gripper_open = True
        self._connected = True

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_busy(self) -> bool:
        return self._busy

    def execute(self, comando: int) -> Dict[str, Any]:
        if self._busy:
            return {"ok": False, "mensaje": "Robot ocupado (rechazado)", "ejecutado": False}
        if comando == 0:
            return self.stop()

        self._busy = True
        if comando in [1, 2, 3]:
            print(f"[MOCK] Moviendo articulación {comando} un paso discreto.")
        elif comando == 4:
            self._gripper_open = not self._gripper_open
            estado = "ABIERTA" if self._gripper_open else "CERRADA"
            print(f"[MOCK] Pinza alternada a estado: {estado}")

        time.sleep(0.3)
        self._busy = False
        return {"ok": True, "mensaje": f"Comando {comando} simulado con éxito", "ejecutado": True}

    def stop(self) -> Dict[str, Any]:
        self._busy = False
        print("[MOCK] Parada de emergencia ejecutada.")
        return {"ok": True, "mensaje": "Parada ejecutada", "ejecutado": True}

    def status(self) -> Dict[str, Any]:
        return {"conectado": self._connected, "busy": self._busy, "gripper_open": self._gripper_open}


class CoppeliaSimAdapter(RobotAdapter):
    """Controlador formal de CoppeliaSim utilizando la API remota ZMQ."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg_robot = cfg.get("robot", {})
        self.step_deg: float = float(self.cfg_robot.get("step_degrees", 15.0))
        self.limits: List[List[float]] = self.cfg_robot.get(
            "joint_limits", [[-170.0, 170.0], [-80.0, 80.0], [-110.0, 110.0]]
        )
        self.tol_rad: float = math.radians(float(self.cfg_robot.get("pos_tolerance_deg", 2.0)))
        self.timeout_s: float = float(self.cfg_robot.get("timeout_s", 3.5))
        self.dist_agarre: float = float(self.cfg_robot.get("dist_agarre", 0.10))

        # Sentido persistente por articulación (+1 o -1)
        self.directions = [1.0, 1.0, 1.0]

        # Estado de la pinza
        self.gripper_open: bool = True
        self.attached_object: Optional[int] = None

        self._busy = False
        self._stop_requested = False
        self._lock = threading.Lock()

        # Handles de objetos en CoppeliaSim
        self.client = None
        self.sim = None
        self.joint_handles = []
        self.gripper_handles = []
        self.tip_handle = -1
        self.graspable_handles = []

        # Posiciones iniciales para reset_scene
        self.initial_joint_positions = []
        self.initial_object_poses = {}

        self.connect()

    def connect(self) -> bool:
        """Establece comunicación con el servidor ZMQ de CoppeliaSim."""
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            self.client = RemoteAPIClient()
            self.sim = self.client.require("sim")

            # Resolver handles de articulaciones principales
            self.joint_handles = [
                self.sim.getObject(name) for name in self.cfg_robot.get("joints", ["/joint1", "/joint2", "/joint3"])
            ]

            # Guardar posiciones iniciales
            self.initial_joint_positions = [self.sim.getJointPosition(h) for h in self.joint_handles]

            # Handles del motor de la pinza
            g_names = self.cfg_robot.get("gripper_joints", ["/PGripStraight/motor"])
            self.gripper_handles = [self.sim.getObject(name) for name in g_names if name]

            # Handle de referencia para calcular proximidad de agarre
            tip_name = self.cfg_robot.get("gripper_tip", "/PGripStraight/body")
            try:
                self.tip_handle = self.sim.getObject(tip_name)
            except Exception:
                self.tip_handle = -1

            # Handles de objetos interactivos
            self.graspable_handles = []
            for obj_name in self.cfg_robot.get("graspable_objects", []):
                try:
                    h = self.sim.getObject(obj_name)
                    self.graspable_handles.append(h)
                    self.initial_object_poses[h] = self.sim.getObjectMatrix(h, -1)
                except Exception:
                    pass

            print("[CoppeliaSimAdapter] Conectado exitosamente vía ZMQ Remote API.")
            return True
        except Exception as e:
            print(f"[CoppeliaSimAdapter] Error de conexión con CoppeliaSim: {e}")
            self.sim = None
            return False

    def disconnect(self) -> None:
        self.sim = None
        self.client = None

    def is_busy(self) -> bool:
        return self._busy

    def stop(self) -> Dict[str, Any]:
        """Detiene el movimiento inmediatamente fijando la posición actual como objetivo."""
        self._stop_requested = True
        if self.sim:
            try:
                for h in self.joint_handles:
                    pos = self.sim.getJointPosition(h)
                    self.sim.setJointTargetPosition(h, pos)
                for gh in self.gripper_handles:
                    self.sim.setJointTargetVelocity(gh, 0.0)
            except Exception as e:
                return {"ok": False, "mensaje": f"Error al detener: {e}", "ejecutado": False}
        self._busy = False
        return {"ok": True, "mensaje": "Movimiento detenido", "ejecutado": True}

    def execute(self, comando: int) -> Dict[str, Any]:
        """Procesa y ejecuta una orden validada."""
        if not self.sim:
            return {"ok": False, "mensaje": "Sin conexión con CoppeliaSim", "ejecutado": False}

        if self._busy:
            return {"ok": False, "mensaje": "Robot ocupado (descartado)", "ejecutado": False}

        if comando == 0:
            return self.stop()

        thread = threading.Thread(target=self._execute_motion, args=(comando,), daemon=True)
        thread.start()
        return {"ok": True, "mensaje": f"Comando {comando} en ejecución", "ejecutado": True}

    def _execute_motion(self, comando: int) -> None:
        with self._lock:
            self._busy = True
            self._stop_requested = False

        try:
            if comando in [1, 2, 3]:
                self._move_joint_discrete(comando - 1)
            elif comando == 4:
                self._toggle_gripper()
        except Exception as e:
            print(f"[CoppeliaSimAdapter] Excepción durante movimiento: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _move_joint_discrete(self, joint_idx: int) -> None:
        """Mueve la articulación joint_idx un paso discreto respetando límites físicos."""
        j_handle = self.joint_handles[joint_idx]
        current_rad = self.sim.getJointPosition(j_handle)
        current_deg = math.degrees(current_rad)

        min_deg, max_deg = self.limits[joint_idx]
        delta_deg = self.step_deg * self.directions[joint_idx]
        target_deg = current_deg + delta_deg

        # Si excede límites mecánicos, invertir sentido persistente
        if target_deg > max_deg or target_deg < min_deg:
            self.directions[joint_idx] *= -1.0
            delta_deg = self.step_deg * self.directions[joint_idx]
            target_deg = current_deg + delta_deg
            target_deg = max(min_deg, min(max_deg, target_deg))

        target_rad = math.radians(target_deg)
        self.sim.setJointTargetPosition(j_handle, target_rad)

        # Espera activa de convergencia con timeout
        t0 = time.time()
        while not self._stop_requested:
            pos_now = self.sim.getJointPosition(j_handle)
            if abs(pos_now - target_rad) <= self.tol_rad:
                break
            if (time.time() - t0) > self.timeout_s:
                print(f"[AVISO] Articulación {joint_idx+1} alcanzó timeout antes de converger.")
                break
            time.sleep(0.01)

    def _toggle_gripper(self) -> None:
        """Alterna apertura/cierre completo de PGripStraight con recorrido extendido."""
        self.gripper_open = not self.gripper_open

        # Velocidad negativa más rápida para cerrar a fondo; positiva para abrir
        vel = 0.12 if self.gripper_open else -0.15
        print(f"[PINZA] {'ABRIENDO' if self.gripper_open else 'CERRANDO A FONDO'} (Velocidad: {vel} m/s)...")

        for gh in self.gripper_handles:
            try:
                # Opcional: garantizar fuerza suficiente en CoppeliaSim
                self.sim.setJointTargetVelocity(gh, vel)
            except Exception as e:
                print(f"[AVISO] Error al fijar velocidad de pinza: {e}")

        # Mayor tiempo de carrera para permitir que las mordazas se junten por completo
        time.sleep(1.8)

        # Frenar el motor una vez cerrado/abierto para no forzar la física
        for gh in self.gripper_handles:
            try:
                # Al cerrar dejamos una mínima fuerza constante de agarre (-0.02)
                vel_mantenimiento = 0.0 if self.gripper_open else -0.02
                self.sim.setJointTargetVelocity(gh, vel_mantenimiento)
            except Exception:
                pass

        if not self.gripper_open:
            self._attempt_grasp()
        else:
            self._release_grasp()
    def _attempt_grasp(self) -> None:
        """Fija como hijo de la pinza el objeto interactivo más cercano si está a dist_agarre."""
        if self.tip_handle == -1:
            return

        tip_pos = self.sim.getObjectPosition(self.tip_handle, -1)
        for obj_h in self.graspable_handles:
            obj_pos = self.sim.getObjectPosition(obj_h, -1)
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(tip_pos, obj_pos)))
            if dist <= self.dist_agarre:
                self.sim.setObjectParent(obj_h, self.tip_handle, True)
                self.attached_object = obj_h
                print(f"[GRASP] Objeto {obj_h} agarrado con éxito.")
                break

    def _release_grasp(self) -> None:
        """Desvincula el objeto agarrado devolviéndolo a la jerarquía del mundo."""
        if self.attached_object is not None:
            self.sim.setObjectParent(self.attached_object, -1, True)
            print(f"[GRASP] Objeto {self.attached_object} liberado.")
            self.attached_object = None

    def reset_scene(self) -> bool:
        """Restaura las articulaciones y los objetos a sus poses iniciales sin reiniciar la CNN."""
        if not self.sim:
            return False
        try:
            self._release_grasp()
            for h, init_pos in zip(self.joint_handles, self.initial_joint_positions):
                self.sim.setJointPosition(h, init_pos)
                self.sim.setJointTargetPosition(h, init_pos)

            for h, init_matrix in self.initial_object_poses.items():
                self.sim.setObjectMatrix(h, -1, init_matrix)

            self.gripper_open = True
            for gh in self.gripper_handles:
                self.sim.setJointTargetVelocity(gh, 0.06)

            print("[CoppeliaSimAdapter] Escena restablecida a la pose inicial.")
            return True
        except Exception as e:
            print(f"[CoppeliaSimAdapter] Error al restablecer escena: {e}")
            return False

    def status(self) -> Dict[str, Any]:
        if not self.sim:
            return {"conectado": False}
        positions = [math.degrees(self.sim.getJointPosition(h)) for h in self.joint_handles]
        return {
            "conectado": True,
            "busy": self._busy,
            "articulaciones_deg": positions,
            "pinza_abierta": self.gripper_open,
            "objeto_agarrado": self.attached_object is not None,
        }