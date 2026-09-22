"""capture.py - Adquisición sistemática de datos crudos (ROI) con metadatos y tasa acotada."""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import yaml

from preprocess import crop_roi, preprocess

CONFIG_PATH = "config.yaml"
LOG_PATH = Path("data/capture_log.csv")
MAX_FPS_CAPTURE = 5.0
MIN_INTERVAL = 1.0 / MAX_FPS_CAPTURE


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Carga los hiperparámetros y configuraciones desde el archivo YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_log_initialized(log_file: Path) -> None:
    """Crea el archivo CSV de auditoría con encabezados si no existe."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        with open(log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "persona",
                "sesion",
                "luz",
                "fondo",
                "fecha_inicio",
                "cam_width",
                "cam_height",
                "roi_x",
                "roi_y",
                "roi_w",
                "roi_h",
            ])


def append_capture_log(
    log_file: Path,
    persona: str,
    sesion: str,
    luz: str,
    fondo: str,
    cam_res: Tuple[int, int],
    roi: Tuple[int, int, int, int],
) -> None:
    """Registra una sesión de captura en el archivo central CSV."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            persona,
            sesion,
            luz,
            fondo,
            now_str,
            cam_res[0],
            cam_res[1],
            roi[0],
            roi[1],
            roi[2],
            roi[3],
        ])


def get_parameters() -> argparse.Namespace:
    """Obtiene los parámetros por CLI o mediante diálogo interactivo si se usa el botón Play."""
    parser = argparse.ArgumentParser(
        description="Captura de datos de gestos para entrenamiento de la CNN."
    )
    parser.add_argument("--persona", type=str, default=None, help="Identificador único (ej: p01)")
    parser.add_argument("--sesion", type=str, default=None, help="Identificador de sesión (ej: s01)")
    parser.add_argument("--luz", type=str, default=None, help="Condición lumínica (ej: natural, blanca, tenue)")
    parser.add_argument("--fondo", type=str, default=None, help="Tipo de fondo (ej: pared_blanca, oficina)")

    args = parser.parse_args()

    # Si se ejecutó con el botón Play sin argumentos, se solicita al usuario por consola
    if args.persona is None or args.sesion is None or args.luz is None or args.fondo is None:
        print("\n=== CONFIGURACIÓN DE LA SESIÓN DE CAPTURA ===")
        print("(Presiona [Enter] para aceptar el valor entre corchetes)\n")

        persona_in = input("ID de Persona [p01]: ").strip()
        args.persona = persona_in if persona_in else "p01"

        sesion_in = input("ID de Sesión [s01]: ").strip()
        args.sesion = sesion_in if sesion_in else "s01"

        luz_in = input("Condición de Luz [natural]: ").strip()
        args.luz = luz_in if luz_in else "natural"

        fondo_in = input("Tipo de Fondo [pared_blanca]: ").strip()
        args.fondo = fondo_in if fondo_in else "pared_blanca"

        print("============================================\n")

    return args


def main() -> None:
    args = get_parameters()
    cfg = load_config(CONFIG_PATH)

    camera_id: int = int(cfg.get("camera_id", 0))
    roi: Tuple[int, int, int, int] = tuple(cfg["roi"])
    rx, ry, rw, rh = roi

    # Crear directorios de salida para data cruda: data/raw/<persona>/<sesion>/<clase>/
    base_save_dir = Path("data") / "raw" / args.persona / args.sesion
    for c in range(5):
        (base_save_dir / str(c)).mkdir(parents=True, exist_ok=True)

    # Contar muestras existentes para no sobrescribir y mantener contadores consistentes
    counts: Dict[int, int] = {}
    for c in range(5):
        class_folder = base_save_dir / str(c)
        existing_files = list(class_folder.glob("*.png"))
        counts[c] = len(existing_files)

    # Inicializar dispositivo de video
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: No se pudo acceder a la cámara {camera_id}.")
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Registrar sesión en log
    ensure_log_initialized(LOG_PATH)
    append_capture_log(
        LOG_PATH,
        args.persona,
        args.sesion,
        args.luz,
        args.fondo,
        (cam_w, cam_h),
        roi,
    )

    print("\n" + "=" * 50)
    print(f" SESIÓN INICIADA: Persona={args.persona} | Sesión={args.sesion}")
    print(f" Almacenando recortes en: {base_save_dir.resolve()}")
    print(" Controles: Mantén presionada la tecla [0, 1, 2, 3, 4] para capturar.")
    print(" Tasa máxima: 5 cuadros/segundo.")
    print(" Presiona [q] o [ESC] para terminar.")
    print("=" * 50 + "\n")

    last_saved_time: float = 0.0
    active_class: Optional[int] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame de la cámara.")
                break

            # Pipeline de preprocesamiento común para validación visual
            _, present, debug = preprocess(frame, cfg)

            # Extraer ROI CRUDO (sin filtros morfológicos ni binarización)
            roi_raw = crop_roi(frame, roi).copy()

            # Captura con límite de tasa temporal (<= 5 FPS)
            current_time = time.time()
            if active_class is not None and (current_time - last_saved_time) >= MIN_INTERVAL:
                timestamp_ms = int(current_time * 1000)
                filename = f"{args.persona}_{args.sesion}_c{active_class}_{timestamp_ms}.png"
                filepath = base_save_dir / str(active_class) / filename
                cv2.imwrite(str(filepath), roi_raw)
                counts[active_class] += 1
                last_saved_time = current_time

            # Dibujar interfaz sobre el frame principal
            display_frame = frame.copy()
            box_color = (0, 255, 0) if present else (0, 0, 255)
            cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), box_color, 2)
            cv2.putText(
                display_frame,
                f"ROI ({'Mano detectada' if present else 'Sin mano'})",
                (rx, max(20, ry - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
                cv2.LINE_AA,
            )

            # Panel informativo con los conteos de la sesión actual
            info_bg = display_frame.copy()
            cv2.rectangle(info_bg, (10, 10), (320, 160), (30, 30, 30), cv2.FILLED)
            display_frame = cv2.addWeighted(info_bg, 0.7, display_frame, 0.3, 0)

            cv2.putText(
                display_frame,
                f"Sujeto: {args.persona} | Sesion: {args.sesion}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            conteo_str = " | ".join([f"C{c}:{counts[c]}" for c in range(5)])
            cv2.putText(
                display_frame,
                conteo_str,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            rec_status = f"GRABANDO -> Clase {active_class}" if active_class is not None else "EN ESPERA"
            rec_color = (0, 0, 255) if active_class is not None else (180, 180, 180)
            cv2.putText(
                display_frame,
                rec_status,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                rec_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                f"Luz: {args.luz} | Fondo: {args.fondo}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Teclas: [0-4] Guardar | [q] Salir",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 255, 150),
                1,
                cv2.LINE_AA,
            )

            # Ventana secundaria con el preprocesamiento (resultado para la CNN)
            prep_vis = cv2.resize(debug["prep"], (150, 150), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Preprocesamiento CNN (64x64)", prep_vis)
            cv2.imshow("Captura de Dataset", display_frame)

            # Gestión de teclas
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in [ord(str(i)) for i in range(5)]:
                active_class = int(chr(key))
            else:
                active_class = None

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCaptura finalizada exitosamente.")
        print(f"Resumen de muestras recolectadas en esta sesion: {counts}\n")


if __name__ == "__main__":
    main()