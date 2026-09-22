"""split_dataset.py - Partición metodológica del dataset por Sujeto o Sesión con auditoría formal."""

import csv
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_RAW_DIR = Path("data/raw")
LOG_CSV_PATH = Path("data/capture_log.csv")
MANIFEST_PATH = Path("data/manifest.csv")
REPORTS_DIR = Path("reports")
CLASSES = ["0", "1", "2", "3", "4"]
SEED = 42

# Proporciones estándar de partición (aprox. 70% / 15% / 15%)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla del generador pseudoaleatorio para reproducibilidad total."""
    random.seed(seed)
    np.random.seed(seed)


def load_metadata_map(log_path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Carga los metadatos de iluminación y fondo desde capture_log.csv."""
    meta_map = {}
    if not log_path.exists():
        return meta_map

    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["persona"], row["sesion"])
            meta_map[key] = {
                "luz": row.get("luz", "desconocida"),
                "fondo": row.get("fondo", "desconocido"),
            }
    return meta_map


def scan_raw_images(raw_dir: Path) -> List[Dict[str, Any]]:
    """Indexa todas las imágenes en data/raw/<persona>/<sesion>/<clase>/."""
    records = []
    if not raw_dir.exists():
        return records

    for persona_dir in sorted(raw_dir.iterdir()):
        if not persona_dir.is_dir():
            continue
        persona = persona_dir.name

        for sesion_dir in sorted(persona_dir.iterdir()):
            if not sesion_dir.is_dir():
                continue
            sesion = sesion_dir.name

            for class_dir in sorted(sesion_dir.iterdir()):
                if not class_dir.is_dir():
                    continue
                clase = class_dir.name
                if clase not in CLASSES:
                    continue

                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                        records.append({
                            "ruta": str(img_path.relative_to(Path("."))),
                            "persona": persona,
                            "sesion": sesion,
                            "clase": clase,
                        })
    return records


def partition_entities(entities: List[str]) -> Dict[str, List[str]]:
    """Distribuye una lista de entidades (personas o sesiones) en train, val y test."""
    shuffled = entities.copy()
    random.shuffle(shuffled)
    n = len(shuffled)

    if n == 1:
        return {"train": shuffled, "val": [], "test": []}
    elif n == 2:
        return {"train": [shuffled[0]], "val": [shuffled[1]], "test": []}
    elif n == 3:
        return {"train": [shuffled[0]], "val": [shuffled[1]], "test": [shuffled[2]]}

    n_val = max(1, int(round(n * SPLIT_RATIOS["val"])))
    n_test = max(1, int(round(n * SPLIT_RATIOS["test"])))
    n_train = n - n_val - n_test

    if n_train <= 0:
        n_train = 1
        n_val = 1
        n_test = n - 2

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def generate_mosaic(manifest_data: List[Dict[str, Any]], out_path: Path) -> None:
    """Genera y guarda un mosaico comparativo con 5 ejemplos por clase e imprime sus rutas."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples_per_class = 5
    fig, axes = plt.subplots(len(CLASSES), samples_per_class, figsize=(10, 10))

    print("\n--- RUTAS DE LAS IMÁGENES DEL MOSAICO ---")
    for row, c in enumerate(CLASSES):
        class_samples = [r for r in manifest_data if r["clase"] == c]
        if len(class_samples) > samples_per_class:
            selected = random.sample(class_samples, samples_per_class)
        else:
            selected = class_samples

        print(f"Clase {c}:")
        for col in range(samples_per_class):
            ax = axes[row, col]
            if col < len(selected):
                ruta = selected[col]["ruta"]
                print(f"  [Col {col+1}] -> {ruta}")
                img_bgr = cv2.imread(ruta)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                ax.set_title(f"C{c} | {selected[col]['split']}", fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"-> Mosaico guardado en: {out_path.resolve()}\n")


def verify_leakage(manifest: List[Dict[str, Any]], unit_key: str) -> bool:
    """Verifica que ninguna persona o sesión esté asignada a múltiples particiones."""
    unit_splits: Dict[str, Set[str]] = {}
    for r in manifest:
        unit = r[unit_key]
        sp = r["split"]
        unit_splits.setdefault(unit, set()).add(sp)

    leakage = False
    for unit, splits in unit_splits.items():
        if len(splits) > 1:
            print(f"[ALERTA CRÍTICA] Fuga detectada en {unit_key} '{unit}': asignado a {splits}")
            leakage = True

    if not leakage:
        print(f"[OK] Comprobación de fuga de datos superada: cero solapamientos por {unit_key}.")
    return not leakage


def main() -> None:
    set_seed(SEED)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    records = scan_raw_images(DATA_RAW_DIR)
    if not records:
        print(f"Error: No se encontraron imágenes válidas en {DATA_RAW_DIR.resolve()}.")
        return

    meta_map = load_metadata_map(LOG_CSV_PATH)
    personas_unicas = sorted(list(set(r["persona"] for r in records)))
    sesiones_unicas = sorted(list(set(f"{r['persona']}::{r['sesion']}" for r in records)))

    print("\n" + "=" * 65)
    print(" PARTICIONAMIENTO METODOLÓGICO DE DATOS (TRAIN / VAL / TEST)")
    print(f" Personas detectadas: {len(personas_unicas)} {personas_unicas}")
    print(f" Sesiones detectadas: {len(sesiones_unicas)}")
    print("=" * 65)

    # Criterio: Particionar por persona si >= 4; por sesión si < 4
    if len(personas_unicas) >= 4:
        criterio = "persona"
        unit_key = "persona"
        partition_map = partition_entities(personas_unicas)
    else:
        criterio = "sesión"
        unit_key = "sesion_id"
        # Asignar identificador compuesto para no colisionar sesiones entre sujetos
        partition_map = partition_entities(sesiones_unicas)

    print(f"-> Estrategia elegida: Partición POR {criterio.upper()} (Semilla={SEED})")
    for sp, ents in partition_map.items():
        print(f"   {sp.upper()} ({len(ents)}): {ents}")

    # Asignar split a cada registro
    split_lookup: Dict[str, str] = {}
    for sp, entities in partition_map.items():
        for e in entities:
            split_lookup[e] = sp

    manifest_rows = []
    for r in records:
        key = r["persona"] if criterio == "persona" else f"{r['persona']}::{r['sesion']}"
        sp = split_lookup.get(key, "train")
        meta = meta_map.get((r["persona"], r["sesion"]), {"luz": "nd", "fondo": "nd"})

        manifest_rows.append({
            "ruta": r["ruta"],
            "persona": r["persona"],
            "sesion": r["sesion"],
            "clase": r["clase"],
            "split": sp,
            "sesion_id": f"{r['persona']}::{r['sesion']}",
            "luz": meta["luz"],
            "fondo": meta["fondo"],
        })

    # Guardar manifest.csv
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ruta", "persona", "sesion", "clase", "split", "luz", "fondo"]
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({
                "ruta": row["ruta"],
                "persona": row["persona"],
                "sesion": row["sesion"],
                "clase": row["clase"],
                "split": row["split"],
                "luz": row["luz"],
                "fondo": row["fondo"],
            })
    print(f"\n-> Manifiesto indexado guardado en: {MANIFEST_PATH.resolve()}")

    # Comprobación de fuga
    verify_leakage(manifest_rows, unit_key)

    # Conteo por clase y por split
    counts_split_class: Dict[str, Dict[str, int]] = {
        s: {c: 0 for c in CLASSES} for s in ["train", "val", "test"]
    }
    for r in manifest_rows:
        counts_split_class[r["split"]][r["clase"]] += 1

    # Conteo por persona y sesión
    counts_persona_sesion: Dict[str, int] = {}
    for r in manifest_rows:
        k = f"{r['persona']} | {r['sesion']} [{r['split']}]"
        counts_persona_sesion[k] = counts_persona_sesion.get(k, 0) + 1

    # Conteo por luz y fondo
    counts_env: Dict[str, int] = {}
    for r in manifest_rows:
        k = f"Luz: {r['luz']} | Fondo: {r['fondo']}"
        counts_env[k] = counts_env.get(k, 0) + 1

    # Generar reporte textual
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(" REPORTE ESTADÍSTICO DE PARTICIÓN Y BALANCE")
    report_lines.append("=" * 60)
    report_lines.append(f"Estrategia de división: Por {criterio}")
    report_lines.append("")
    report_lines.append("1. DISTRIBUCIÓN POR CLASE Y SPLIT:")
    report_lines.append(f"{'Clase':<8} | {'Train':<10} | {'Val':<10} | {'Test':<10}")
    report_lines.append("-" * 44)
    for c in CLASSES:
        tr = counts_split_class["train"][c]
        va = counts_split_class["val"][c]
        te = counts_split_class["test"][c]
        report_lines.append(f"Clase {c:<2} | {tr:<10} | {va:<10} | {te:<10}")
    report_lines.append("-" * 44)
    tot_tr = sum(counts_split_class["train"].values())
    tot_va = sum(counts_split_class["val"].values())
    tot_te = sum(counts_split_class["test"].values())
    report_lines.append(f"{'TOTAL':<8} | {tot_tr:<10} | {tot_va:<10} | {tot_te:<10}")
    report_lines.append("")

    report_lines.append("2. DISTRIBUCIÓN POR PERSONA Y SESIÓN:")
    for ps, count in sorted(counts_persona_sesion.items()):
        report_lines.append(f"  {ps}: {count} imágenes")
    report_lines.append("")

    report_lines.append("3. DISTRIBUCIÓN POR CONDICIÓN AMBIENTAL (LUZ / FONDO):")
    for env, count in sorted(counts_env.items()):
        report_lines.append(f"  {env}: {count} imágenes")
    report_lines.append("=" * 60)

    report_str = "\n".join(report_lines)
    print("\n" + report_str)

    with open(REPORTS_DIR / "split_report.txt", "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"-> Reporte textual guardado en: {(REPORTS_DIR / 'split_report.txt').resolve()}")

    # Alertas de desbalance o escasez
    print("\n=== AUDITORÍA DE BALANCE DE CLASES ===")
    for sp in ["train", "val", "test"]:
        sub_counts = [counts_split_class[sp][c] for c in CLASSES]
        if not sub_counts or sum(sub_counts) == 0:
            print(f"[AVISO] El split '{sp}' está vacío.")
            continue

        for c in CLASSES:
            n_c = counts_split_class[sp][c]
            if n_c < 50:
                print(f"[ALERTA DE ESCASEZ] Split '{sp}', Clase {c} tiene solo {n_c} imágenes (< 50).")

        c_min = min(sub_counts)
        c_max = max(sub_counts)
        if c_min > 0:
            ratio = c_max / c_min
            if ratio > 3.0:
                print(
                    f"[ALERTA DESBALANCE] Split '{sp}' presenta un ratio de desbalance {ratio:.2f}:1 (> 3:1) "
                    f"(Max Clase: {c_max}, Min Clase: {c_min})."
                )
        else:
            print(f"[ALERTA DESBALANCE CRÍTICA] Split '{sp}' tiene clases con 0 muestras.")

    # Generar mosaico visual de validación
    generate_mosaic(manifest_rows, REPORTS_DIR / "mosaico_clases.png")
    print("\nFase 3 finalizada correctamente.\n")


if __name__ == "__main__":
    main()