"""run_sequences.py - Validación autónoma de secuencias robóticas con desbloqueo guiado de FSM."""

import csv
from datetime import datetime
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import cv2
import yaml

from cnn_inference import GestureClassifier
from command_filter import CommandFilter
from robot_adapter import CoppeliaSimAdapter

CONFIG_PATH = "config.yaml"
REPORTS_DIR = Path("reports")
SEQUENCES_LOG_CSV = REPORTS_DIR / "phase8_secuencias.csv"

# Secuencia con tiempos holgados (8 segundos por paso)
TASK_SEQUENCES = {
    "Secuencia_PickAndPlace": [
        (1, 8.0, "Girar base hacia objeto (Clase 1)"),
        (2, 8.0, "Bajar hombro (Clase 2)"),
        (4, 8.0, "Cerrar pinza sobre cilindro (Clase 4)"),
        (2, 8.0, "Levantar hombro (Clase 2)"),
        (1, 8.0, "Girar hacia zona deposito (Clase 1)"),
        (4, 8.0, "Abrir pinza y soltar (Clase 4)"),
    ]
}


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_deposit_success(
    sim: Any,
    obj_name: str,
    drop_zone_name: str = "/drop_zone",
    max_dist: float = 0.18,
) -> bool:
    """Verifica si el objeto está desvinculado de la pinza y dentro del radio de depósito."""
    try:
        obj_handle = sim.getObject(obj_name)
        zone_handle = sim.getObject(drop_zone_name)
    except Exception:
        return False

    parent = sim.getObjectParent(obj_handle)
    if parent != -1:
        return False

    p_obj = sim.getObjectPosition(obj_handle, -1)
    p_zone = sim.getObjectPosition(zone_handle, -1)
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p_obj, p_zone)))
    return dist <= max_dist


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    adapter = CoppeliaSimAdapter(cfg)
    if not adapter.sim:
        print("Error: No se pudo conectar con CoppeliaSim vía ZMQ.")
        return

    classifier = GestureClassifier(CONFIG_PATH)
    filter_engine = CommandFilter(cfg)

    # Crear zona de depósito si no existe
    try:
        drop_zone_handle = adapter.sim.getObject("/drop_zone")
    except Exception:
        drop_zone_handle = adapter.sim.createDummy(0.05)
        adapter.sim.setObjectAlias(drop_zone_handle, "drop_zone")
        adapter.sim.setObjectPosition(drop_zone_handle, -1, [0.25, 0.25, 0.05])

    cap = cv2.VideoCapture(int(cfg.get("camera_id", 0)))
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    rx, ry, rw, rh = tuple(cfg["roi"])
    win_name = "Secuencias Roboticas - Evaluacion Autonoma"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    results = []

    print("\n" + "=" * 70)
    print(" INICIANDO SECUENCIA GUIADA (Sigue las instrucciones en la ventana)")
    print("=" * 70 + "\n")

    try:
        for seq_name, seq_steps in TASK_SEQUENCES.items():
            print(f"\n>>>> Tarea: {seq_name} ({len(seq_steps)} pasos) <<<<")
            adapter.reset_scene()
            time.sleep(1.0)

            all_perceptions_ok = True
            all_commands_ok = True

            for step_idx, (expected_cmd, step_duration, step_desc) in enumerate(seq_steps, 1):
                # ---------------- TRANSICIÓN: OBLIGAR A RETIRAR LA MANO ----------------
                # Esto garantiza que el filtro pase de WAIT_ZERO a IDLE de forma limpia
                filter_engine.reset_to_idle()
                t_trans = time.time()
                while (time.time() - t_trans) < 1.2:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    if cfg.get("flip_horizontal", True):
                        frame = cv2.flip(frame, 1)

                    disp = frame.copy()
                    cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 165, 255), 2)
                    cv2.putText(disp, "PREPARANDO SIGUIENTE PASO...", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(disp, "Retira la mano o pon Clase 0", (30, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow(win_name, disp)
                    cv2.waitKey(1)

                # ---------------- EJECUCIÓN ACTIVA DEL PASO ----------------
                t_start = time.time()
                step_perceived_ok = False
                step_command_fired = False

                while (time.time() - t_start) < step_duration:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    if cfg.get("flip_horizontal", True):
                        frame = cv2.flip(frame, 1)

                    now_s = time.time()
                    rem_time = step_duration - (now_s - t_start)

                    # Inferencia
                    inf = classifier.predict(frame)
                    raw_pred = inf["clase"]
                    conf = inf["confianza"]
                    present = inf["hand_present"]

                    if raw_pred == expected_cmd and conf >= cfg.get("confidence_threshold", 0.85):
                        step_perceived_ok = True

                    # Filtro de Seguridad
                    filter_engine.set_robot_busy(adapter.is_busy())
                    filt = filter_engine.update(raw_pred, conf, now_s, present)

                    # Disparo
                    if filt["comando_aceptado"] == expected_cmd:
                        step_command_fired = True
                        print(f" -> [PASO {step_idx} ACEPTADO] Ejecutando comando {expected_cmd}...")
                        adapter.execute(expected_cmd)
                        # Esperar fin de movimiento
                        while adapter.is_busy():
                            time.sleep(0.05)
                        break

                    # Dibujo de GUI
                    disp = frame.copy()
                    box_color = (0, 255, 0) if present else (0, 0, 255)
                    cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), box_color, 2)

                    overlay = disp.copy()
                    cv2.rectangle(overlay, (10, 10), (480, 130), (20, 20, 20), -1)
                    disp = cv2.addWeighted(overlay, 0.75, disp, 0.25, 0)

                    cv2.putText(disp, f"PASO {step_idx}/{len(seq_steps)}: {step_desc}", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    cv2.putText(disp, f"Muestra: {expected_cmd} DEDOS | Tiempo restante: {rem_time:.1f}s", (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.putText(disp, f"Pred: C{raw_pred} ({conf*100:.0f}%) | Estado: {filt['estado']}", (20, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.putText(disp, f"Rechazo: {filt['motivo_rechazo'] if filt['motivo_rechazo'] else 'Ninguno'}", (20, 118),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)

                    cv2.imshow(win_name, disp)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        return

                if not step_perceived_ok:
                    all_perceptions_ok = False
                if not step_command_fired:
                    all_commands_ok = False

            # Evaluación física automática vía API
            target_obj = cfg["robot"].get("graspable_objects", ["/Cylinder"])[0]
            execution_ok = check_deposit_success(adapter.sim, target_obj)

            print(f"\n--- RESUMEN DE LA TAREA [{seq_name}] ---")
            print(f" -> Éxito de Percepción (CNN)  : {'SI' if all_perceptions_ok else 'NO'}")
            print(f" -> Éxito de Comando (Filtro)  : {'SI' if all_commands_ok else 'NO'}")
            print(f" -> Éxito de Ejecución Física  : {'SI' if execution_ok else 'NO'}")

            results.append({
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "secuencia": seq_name,
                "percepcion_exitosa": all_perceptions_ok,
                "comando_exitoso": all_commands_ok,
                "ejecucion_fisica_exitosa": execution_ok,
            })

    finally:
        cap.release()
        cv2.destroyAllWindows()
        adapter.disconnect()

    # Guardar en CSV
    with open(SEQUENCES_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fecha", "secuencia", "percepcion_exitosa", "comando_exitoso", "ejecucion_fisica_exitosa"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[OK] Resultados guardados en: {SEQUENCES_LOG_CSV.resolve()}\n")


if __name__ == "__main__":
    main()