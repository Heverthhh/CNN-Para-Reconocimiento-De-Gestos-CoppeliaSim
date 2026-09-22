"""preprocess.py - Preprocesamiento COMÚN a captura, entrenamiento e inferencia."""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def crop_roi(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Extrae la sub-imagen delimitada por la región de interés (x, y, w, h)."""
    x, y, w, h = roi
    return frame_bgr[y:y + h, x:x + w]


def skin_mask(
    roi_bgr: np.ndarray,
    cr: Tuple[int, int] = (133, 173),
    cb: Tuple[int, int] = (77, 127)
) -> np.ndarray:
    """Calcula la máscara binaria de piel en espacio YCrCb aplicando morfología."""
    blur = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
    ycc = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycc, (0, int(cr[0]), int(cb[0])), (255, int(cr[1]), int(cb[1])))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
    return mask


def _largest_component(mask: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Filtra y devuelve únicamente el contorno y la máscara del componente más grande."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, np.zeros_like(mask)
    c = max(cnts, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [c], -1, 255, thickness=cv2.FILLED)
    return c, clean


def _square_crop(img: np.ndarray, box: Tuple[int, int, int, int], pad: float = 0.10) -> np.ndarray:
    """Genera un recorte cuadrado centrado con relleno perimetral para invariancia de escala."""
    x, y, w, h = box
    side = int(max(w, h) * (1 + 2 * pad))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = cx - side // 2, cy - side // 2
    H, W = img.shape[:2]
    canvas = np.zeros((side, side), dtype=img.dtype)
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x0 + side, W), min(y0 + side, H)
    canvas[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = img[ys0:ys1, xs0:xs1]
    return canvas


def preprocess(
    frame_bgr: np.ndarray,
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, bool, Dict[str, np.ndarray]]:
    """Aplica el canal de preprocesamiento estándar a un frame de video.
    
    Args:
        frame_bgr: Imagen BGR de entrada capturada por la cámara.
        cfg: Diccionario con llaves 'roi', 'img_size', 'mode', 'cr', 'cb', 'min_area_frac'.

    Returns:
        x: Tensor numpy float32 (1, S, S) con valores en [0, 1].
        present: Booleano que indica presencia de mano válida dentro del ROI.
        debug: Diccionario con imágenes intermedias ('roi', 'mask', 'prep').
    """
    S: int = int(cfg["img_size"])
    roi_coords: Tuple[int, int, int, int] = tuple(cfg["roi"])
    roi = crop_roi(frame_bgr, roi_coords)

    gray = _CLAHE.apply(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    cr = tuple(cfg.get("cr", (133, 173)))
    cb = tuple(cfg.get("cb", (77, 127)))
    mask = skin_mask(roi, cr, cb)
    contour, mask = _largest_component(mask)

    area_frac = float(mask.sum() / 255) / float(mask.size) if mask.size > 0 else 0.0
    min_area_frac = float(cfg.get("min_area_frac", 0.04))
    present = contour is not None and area_frac >= min_area_frac

    mode: str = cfg["mode"]
    if mode == "gray":
        base = gray
    elif mode == "mask":
        base = mask
    elif mode == "masked_gray":
        base = cv2.bitwise_and(gray, gray, mask=mask)
    else:
        raise ValueError(f"mode desconocido: {mode}")

    if mode != "gray" and present and contour is not None:
        base = _square_crop(base, cv2.boundingRect(contour))

    out = cv2.resize(base, (S, S), interpolation=cv2.INTER_AREA)
    x = (out.astype(np.float32) / 255.0)[None, ...]
    return x, present, {"roi": roi, "mask": mask, "prep": out}