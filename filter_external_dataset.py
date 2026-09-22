"""filter_external_dataset.py - Audita y filtra un dataset existente contra preprocess.py."""

from pathlib import Path
import shutil
from typing import Any, Dict, Tuple
import cv2
import yaml

from preprocess import preprocess

CONFIG_PATH = "config.yaml"
# Ruta donde está tu dataset descargado actualmente
SOURCE_DIR = Path("dataset")
# Destino estandarizado del proyecto
DEST_DIR = Path("data/raw")


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Carga hiperparámetros desde config.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def audit_and_import() -> None:
    """Filtra imágenes incompatibles con el preprocesamiento y las estructura en data/raw."""
    if not SOURCE_DIR.exists():
        print(f"Error: No se encontró la carpeta de origen '{SOURCE_DIR.resolve()}'.")
        print("Asegúrate de colocar la carpeta con nombre 'dataset' en la raíz del proyecto.")
        return

    cfg = load_config(CONFIG_PATH)

    subsets = ["train", "val", "test"]
    clases = ["0", "1", "2", "3", "4"]

    stats = {
        split: {c: {"aceptadas": 0, "rechazadas": 0} for c in clases}
        for split in subsets
    }

    print("\n" + "=" * 60)
    print(" INICIANDO AUDITORÍA Y FILTRADO DEL DATASET EXTERNO")
    print(f" Origen:  {SOURCE_DIR.resolve()}")
    print(f" Destino: {DEST_DIR.resolve()}")
    print(" Criterio: Compatibilidad directa con skin_mask y min_area_frac")
    print("=" * 60 + "\n")

    for split in subsets:
        split_path = SOURCE_DIR / split
        if not split_path.exists():
            continue

        # Mapeamos cada split a un ID de sesión para no romper la trazabilidad
        session_id = f"external_{split}"
        persona_id = "p_ext"

        for c in clases:
            class_folder = split_path / c
            if not class_folder.exists():
                continue

            target_folder = DEST_DIR / persona_id / session_id / c
            target_folder.mkdir(parents=True, exist_ok=True)

            image_files = [
                p for p in class_folder.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
            ]

            for img_path in image_files:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    stats[split][c]["rechazadas"] += 1
                    continue

                # Si la imagen ya es un recorte de la mano, adaptamos un ROI relativo al tamaño total
                # para que preprocess.py funcione idénticamente sin recortar fuera de los límites
                h, w = img_bgr.shape[:2]
                local_cfg = cfg.copy()
                local_cfg["roi"] = [0, 0, w, h]

                # Pasamos la imagen por el pipeline real
                try:
                    x, present, debug = preprocess(img_bgr, local_cfg)
                except Exception:
                    stats[split][c]["rechazadas"] += 1
                    continue

                # Criterio de validación técnica:
                # - Clases 1 a 4: DEBEN tener mano detectable por piel y morfología
                # - Clase 0: Si tu clase 0 es puño cerrado, requiere present == True.
                #            Si tu clase 0 es fondo vacío, no debe detectar nada.
                # Como estándar general de clasificación gestual (0 = puño cerrado):
                valida = True
                if c in ["1", "2", "3", "4"]:
                    if not present:
                        valida = False

                if valida:
                    # Copiamos la imagen limpia a la estructura del laboratorio
                    dest_file = target_folder / f"{persona_id}_{session_id}_c{c}_{img_path.name}"
                    shutil.copy(img_path, dest_file)
                    stats[split][c]["aceptadas"] += 1
                else:
                    stats[split][c]["rechazadas"] += 1

    # Reporte de resultados
    print("\n=== RESUMEN DE COMPATIBILIDAD ===")
    total_in = 0
    total_out = 0
    for split in subsets:
        print(f"\n--- Subconjunto: {split} ---")
        for c in clases:
            acc = stats[split][c]["aceptadas"]
            rej = stats[split][c]["rechazadas"]
            total_in += acc
            total_out += rej
            t_class = acc + rej
            pct = (acc / t_class * 100.0) if t_class > 0 else 0.0
            print(f"  Clase {c}: {acc} aceptadas, {rej} descartadas ({pct:.1f}% útiles)")

    total = total_in + total_out
    global_pct = (total_in / total * 100.0) if total > 0 else 0.0
    print("\n" + "=" * 60)
    print(f" TOTAL EVALUADO: {total} imágenes")
    print(f" IMÁGENES COMPATIBLES GUARDADAS: {total_in} ({global_pct:.1f}%)")
    print(f" IMÁGENES DESCARTADAS: {total_out}")
    print(f" Destino final: {DEST_DIR.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    audit_and_import()