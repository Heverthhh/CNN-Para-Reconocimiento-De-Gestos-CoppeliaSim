"""record_trials.py - Grabación automatizada de sesiones de prueba etiquetadas con metadatos."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict

import cv2
import yaml

CONFIG_PATH = "config.yaml"
VALID_CONDITIONS = [
    "normal",
    "luz_baja",
    "luz_alta",
    "fondo_complejo",
    "mano_parcial",
    "transicion_rapida",
    "sin_mano",
]


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grabación de videos de prueba para Fase 8.")
    parser.add_argument("--clase", type=int, required=True, choices=[0, 1, 2, 3, 4], help="Clase real del gesto (0..4).")
    parser.add_argument("--condicion", type=str, required=True, choices=VALID_CONDITIONS, help="Condición ambiental.")
    parser.add_argument("--persona", type=str, required=True, help="Identificador del sujeto (ej: p01).")
    parser.add_argument("--duracion_s", type=float, default=15.0, help="Duración del ensayo en segundos.")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    camera_id = int(cfg.get("camera_id", 0))

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {camera_id}.")
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_nominal = cap.get(cv2.CAP_PROP_FPS)
    if fps_nominal <= 0 or fps_nominal > 120:
        fps_nominal = 30.0

    # Crear directorio: data/trials/<persona>/<condicion>/
    save_dir = Path("data") / "trials" / args.persona / args.condicion
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = save_dir / f"clase_{args.clase}_{timestamp_str}.mp4"
    meta_path = save_dir / f"clase_{args.clase}_{timestamp_str}.json"

    rx, ry, rw, rh = tuple(cfg["roi"])
    win_name = "Grabacion de Ensayo - Fase 8"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # 1. Cuenta regresiva interactiva de 3 segundos
    t_start_cd = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cfg.get("flip_horizontal", True):
            frame = cv2.flip(frame, 1)

        elapsed = time.time() - t_start_cd
        remaining = 3.0 - elapsed

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        cv2.putText(frame, f"PREPARATE: Clase {args.clase} [{args.condicion}]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if remaining > 0:
            cv2.putText(frame, f"INICIO EN: {int(remaining) + 1}", (int(cam_w / 2) - 100, int(cam_h / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        else:
            break

        cv2.imshow(win_name, frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            cap.release()
            cv2.destroyAllWindows()
            return

    # 2. Grabación activa de video crudo (sin alterar píxeles)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps_nominal, (cam_w, cam_h))

    frames_recorded = 0
    t_record_start = time.time()

    print(f"\n[GRABANDO] Guardando en: {video_path.name} ({args.duracion_s} s)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cfg.get("flip_horizontal", True):
            frame = cv2.flip(frame, 1)

        t_cur = time.time()
        dur_now = t_cur - t_record_start
        if dur_now >= args.duracion_s:
            break

        out.write(frame)
        frames_recorded += 1

        # Interfaz visual
        disp = frame.copy()
        cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.circle(disp, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(disp, f"REC ({args.duracion_s - dur_now:.1f}s) | C{args.clase}", (50, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.imshow(win_name, disp)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    t_total = time.time() - t_record_start
    fps_real = frames_recorded / t_total if t_total > 0 else fps_nominal

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 3. Almacenamiento del JSON de metadatos
    metadata = {
        "clase_real": int(args.clase),
        "condicion": args.condicion,
        "persona": args.persona,
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fps_estimado": round(fps_real, 2),
        "total_cuadros": frames_recorded,
        "duracion_segundos": round(t_total, 2),
        "resolucion": [cam_w, cam_h],
        "roi": cfg.get("roi"),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"[OK] Video crudo guardado: {video_path.resolve()}")
    print(f"[OK] Metadatos guardados: {meta_path.resolve()}\n")


if __name__ == "__main__":
    main()