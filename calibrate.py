"""calibrate.py - Herramienta interactiva para sintonizar el preprocesamiento en tiempo real."""

from typing import Any, Dict, List
import cv2
import numpy as np
import yaml

from preprocess import preprocess

CONFIG_PATH = "config.yaml"
MODES: List[str] = ["mask", "masked_gray", "gray"]


def _nothing(_: int) -> None:
    pass


def load_initial_config(path: str) -> Dict[str, Any]:
    """Carga configuración base o genera valores predeterminados seguros."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "camera_id": 0,
            "roi": [350, 100, 260, 260],
            "img_size": 64,
            "mode": "mask",
            "cr": [133, 173],
            "cb": [77, 127],
            "min_area_frac": 0.04,
        }


def main() -> None:
    cfg = load_initial_config(CONFIG_PATH)
    mode_idx = MODES.index(cfg.get("mode", "mask")) if cfg.get("mode") in MODES else 0

    win_name = "Calibracion de Preprocesamiento (q: salir, s: guardar YAML, m: alternar modo)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # Crear trackbars para YCrCb y área mínima
    cv2.createTrackbar("Cr Min", win_name, int(cfg["cr"][0]), 255, _nothing)
    cv2.createTrackbar("Cr Max", win_name, int(cfg["cr"][1]), 255, _nothing)
    cv2.createTrackbar("Cb Min", win_name, int(cfg["cb"][0]), 255, _nothing)
    cv2.createTrackbar("Cb Max", win_name, int(cfg["cb"][1]), 255, _nothing)
    # Rango 0 a 500 equivale a 0.000 hasta 0.500 de fracción de área
    initial_area_int = int(cfg.get("min_area_frac", 0.04) * 1000)
    cv2.createTrackbar("Min Area (x1000)", win_name, initial_area_int, 500, _nothing)

    cap = cv2.VideoCapture(int(cfg.get("camera_id", 0)))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {cfg.get('camera_id', 0)}")
        return

    print("\n--- CONTROLES ---")
    print(" [m] Alternar modo ('mask' -> 'masked_gray' -> 'gray')")
    print(" [s] Imprimir parámetros actuales en formato YAML")
    print(" [q] o [ESC] Salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar cuadro de video.")
            break

        # Sincronizar parámetros con los trackbars
        cr_min = cv2.getTrackbarPos("Cr Min", win_name)
        cr_max = cv2.getTrackbarPos("Cr Max", win_name)
        cb_min = cv2.getTrackbarPos("Cb Min", win_name)
        cb_max = cv2.getTrackbarPos("Cb Max", win_name)
        area_val = cv2.getTrackbarPos("Min Area (x1000)", win_name) / 1000.0

        cfg["cr"] = [cr_min, cr_max]
        cfg["cb"] = [cb_min, cb_max]
        cfg["min_area_frac"] = float(area_val)
        cfg["mode"] = MODES[mode_idx]

        # Inferencia de preprocesamiento estándar
        _, present, debug = preprocess(frame, cfg)

        roi_img = debug["roi"]
        mask_bgr = cv2.cvtColor(debug["mask"], cv2.COLOR_GRAY2BGR)
        prep_bgr = cv2.cvtColor(debug["prep"], cv2.COLOR_GRAY2BGR)

        # Escalar visualización a tamaño comparable con el ROI
        h_roi, w_roi = roi_img.shape[:2]
        mask_view = cv2.resize(mask_bgr, (w_roi, h_roi))
        prep_view = cv2.resize(prep_bgr, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)

        # Panel lado a lado: [ROI | MÁSCARA | SALIDA PREP]
        combined_view = np.hstack([roi_img, mask_view, prep_view])

        # Estado visual
        color_status = (0, 255, 0) if present else (0, 0, 255)
        text_status = f"Presente: {present} | Modo: {cfg['mode']} | MinArea: {area_val:.3f}"
        cv2.putText(
            combined_view,
            text_status,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_status,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(win_name, combined_view)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("m"):
            mode_idx = (mode_idx + 1) % len(MODES)
            print(f"-> Modo cambiado a: {MODES[mode_idx]}")
        elif key == ord("s"):
            yaml_out = {
                "roi": cfg["roi"],
                "img_size": cfg["img_size"],
                "mode": cfg["mode"],
                "cr": [cr_min, cr_max],
                "cb": [cb_min, cb_max],
                "min_area_frac": float(area_val),
            }
            print("\n" + "=" * 40)
            print("# COPIA ESTA SECCIÓN A TU config.yaml:")
            print(yaml.dump(yaml_out, default_flow_style=None, sort_keys=False))
            print("=" * 40 + "\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()