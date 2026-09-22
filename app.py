"""app.py - Interfaz en tiempo real con efecto espejo, telemetría y control robótico seguro."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import yaml

from cnn_inference import GestureClassifier
from command_filter import CommandFilter

CONFIG_PATH = "config.yaml"
LOGS_DIR = Path("logs")
CLASSES = [0, 1, 2, 3, 4]
CLASS_NAMES = ["0: Parada", "1: Art 1", "2: Art 2", "3: Art 3", "4: Pinza"]


class MockRobotAdapter:
    """Adaptador simulado para ensayos en seco cuando no se dispone de CoppeliaSim."""

    def __init__(self) -> None:
        self.connected = True

    def execute(self, cmd: int) -> Dict[str, Any]:
        print(f"[MOCK ROBOT] Acción ejecutada con éxito -> Comando: {CLASS_NAMES[cmd]}")
        return {"ok": True, "mensaje": "Simulado", "ejecutado": True}

    def is_busy(self) -> bool:
        return False


def draw_probability_bars(canvas: np.ndarray, probs: list, x_start: int, y_start: int) -> None:
    """Dibuja barras horizontales de probabilidad para cada clase."""
    bar_width = 120
    bar_height = 12
    gap = 6

    for i, p in enumerate(probs):
        y = y_start + i * (bar_height + gap)
        cv2.putText(
            canvas,
            f"C{i}:",
            (x_start, y + bar_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (x_start + 30, y), (x_start + 30 + bar_width, y + bar_height), (50, 50, 50), -1)
        fill_w = int(bar_width * p)
        color = (0, 255, 0) if p == max(probs) else (255, 165, 0)
        cv2.rectangle(canvas, (x_start + 30, y), (x_start + 30 + fill_w, y + bar_height), color, -1)
        cv2.putText(
            canvas,
            f"{p*100:.0f}%",
            (x_start + 35 + bar_width, y + bar_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aplicación interactiva de control gestual.")
    parser.add_argument("--sin-robot", action="store_true", help="Usa MockAdapter para ensayos en seco.")
    parser.add_argument("--persona", type=str, default="operador_01", help="ID del operador para el log.")
    parser.add_argument("--condicion", type=str, default="laboratorio", help="Condición ambiental para el log.")
    args = parser.parse_args()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    classifier = GestureClassifier(CONFIG_PATH)
    filter_engine = CommandFilter(cfg)

    # Conexión con Robot (o Mock)
    if args.sin_robot:
        robot = MockRobotAdapter()
        robot_status = "Mock Robot (En Seco)"
    else:
        try:
            from robot_adapter import CoppeliaSimAdapter
            robot = CoppeliaSimAdapter(cfg)
            if robot.sim is not None:
                robot_status = "Conectado a CoppeliaSim"
            else:
                robot = None
                robot_status = "Sin conexion con CoppeliaSim"
        except Exception:
            robot = None
            robot_status = "Sin robot"

    cap = cv2.VideoCapture(int(cfg.get("camera_id", 0)))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {cfg.get('camera_id', 0)}.")
        return

    rx, ry, rw, rh = tuple(cfg["roi"])
    logging_active = False
    log_file: Optional[Path] = None
    csv_writer = None
    csv_handle = None
    ensayo_class_ground_truth: int = 0

    last_accepted_cmd: Optional[int] = None
    win_name = "Control Gestual - CNN & Seguridad"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 65)
    print(f" SISTEMA DE CONTROL GESTUAL INICIADO [{robot_status.upper()}]")
    print(" Controles de teclado:")
    print("   [p] PARADA forzada (Clase 0)")
    print("   [r] Reiniciar filtro a IDLE")
    print("   [l] Activar/Desactivar Logging a CSV")
    print("   [0..4] Marcar clase real del ensayo actual")
    print("   [q] o [ESC] Salir")
    print("=" * 65 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Modo espejo si está activado
            if cfg.get("flip_horizontal", True):
                frame = cv2.flip(frame, 1)

            now_s = time.time()

            # 1. Inferencia CNN
            inf_res = classifier.predict(frame)
            raw_pred = inf_res["clase"]
            conf = inf_res["confianza"]
            hand_present = inf_res["hand_present"]
            lat = inf_res["latencia_ms"]
            debug = inf_res["debug"]

            # 2. Informar al filtro si el robot físico/simulado está ocupado
            if robot and hasattr(robot, "is_busy"):
                filter_engine.set_robot_busy(robot.is_busy())

            # 3. Filtrado de Seguridad FSM
            filt_res = filter_engine.update(raw_pred, conf, now_s, hand_present)
            accepted = filt_res["comando_aceptado"]
            fsm_state = filt_res["estado"]
            reject_reason = filt_res["motivo_rechazo"]

            # 4. Enviar a Robot si fue formalmente aceptado por el filtro
            if accepted is not None:
                last_accepted_cmd = accepted
                if robot:
                    robot.execute(accepted)

            # 5. Registro en log CSV
            if logging_active and csv_writer is not None:
                csv_writer.writerow([
                    round(now_s, 4),
                    ensayo_class_ground_truth,
                    raw_pred,
                    round(conf, 4),
                    accepted if accepted is not None else "",
                    round(lat["total"], 2),
                    reject_reason if reject_reason else "",
                    args.persona,
                    args.condicion,
                ])

            # 6. Construcción gráfica de la Interfaz
            display = frame.copy()
            roi_color = (0, 255, 0) if hand_present else (0, 0, 255)
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), roi_color, 2)

            # Panel de Telemetría Superior Izquierdo
            overlay = display.copy()
            cv2.rectangle(overlay, (10, 10), (320, 230), (20, 20, 20), -1)
            display = cv2.addWeighted(overlay, 0.75, display, 0.25, 0)

            cv2.putText(display, f"Estado Robot: {robot_status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display, f"FSM Filtro: {fsm_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255) if fsm_state == "IDLE" else (0, 100, 255), 2, cv2.LINE_AA)

            # Predicción y confianza
            cv2.putText(
                display,
                f"Pred: C{raw_pred} ({conf*100:.1f}%)",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if conf >= cfg.get("confidence_threshold", 0.85) else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(display, f"Rechazo: {reject_reason if reject_reason else 'Ninguno'}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            # Comando aceptado
            cmd_text = f"Comando: {CLASS_NAMES[last_accepted_cmd]}" if last_accepted_cmd is not None else "Comando: Ninguno"
            cv2.putText(display, cmd_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            # Latencias
            cv2.putText(
                display,
                f"Lat: P:{lat['preproc']:.1f}ms | C:{lat['cnn']:.1f}ms | T:{lat['total']:.1f}ms",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Estado del Log
            log_str = f"LOG: ACTIVO (GT=C{ensayo_class_ground_truth})" if logging_active else "LOG: INACTIVO"
            cv2.putText(display, log_str, (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if logging_active else (150, 150, 150), 1, cv2.LINE_AA)

            # Barras de probabilidad Softmax
            draw_probability_bars(display, inf_res["probs"], x_start=20, y_start=180)

            # Entrada a la CNN (64x64) en la esquina inferior derecha
            prep_vis = cv2.resize(debug["prep"], (110, 110), interpolation=cv2.INTER_NEAREST)
            prep_bgr = cv2.cvtColor(prep_vis, cv2.COLOR_GRAY2BGR)
            h_f, w_f = display.shape[:2]
            display[h_f - 120:h_f - 10, w_f - 120:w_f - 10] = prep_bgr
            cv2.rectangle(display, (w_f - 120, h_f - 120), (w_f - 10, h_f - 10), (255, 255, 255), 1)

            cv2.imshow(win_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("p"):
                filter_engine.reset_to_idle()
                last_accepted_cmd = 0
                if robot:
                    robot.execute(0)
                print("[SEGURIDAD] PARADA manual invocada ('p').")
            elif key == ord("r"):
                filter_engine.reset_to_idle()
                print("[SISTEMA] Filtro reiniciado a IDLE ('r').")
            elif key == ord("l"):
                logging_active = not logging_active
                if logging_active:
                    LOGS_DIR.mkdir(parents=True, exist_ok=True)
                    log_file = LOGS_DIR / f"ensayos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    csv_handle = open(log_file, "w", newline="", encoding="utf-8")
                    csv_writer = csv.writer(csv_handle)
                    csv_writer.writerow([
                        "t",
                        "clase_real",
                        "pred_cruda",
                        "confianza",
                        "comando_aceptado",
                        "latencia_ms",
                        "motivo_rechazo",
                        "persona",
                        "condicion",
                    ])
                    print(f"[LOG] Grabación activada: {log_file.name}")
                else:
                    if csv_writer and csv_handle:
                        csv_handle.close()
                        csv_writer = None
                        csv_handle = None
                    print("[LOG] Grabación pausada.")
            elif key in [ord(str(i)) for i in range(5)]:
                ensayo_class_ground_truth = int(chr(key))
                print(f"[MODO ENSAYO] Clase real fijada en: {ensayo_class_ground_truth}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if logging_active and csv_writer and csv_handle:
            csv_handle.close()
        print("\nAplicación cerrada correctamente.")


if __name__ == "__main__":
    main()