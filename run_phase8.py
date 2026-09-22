"""run_phase8.py - Evaluación de robustez, diagnóstico causal y optimización del filtro."""

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from cnn_inference import GestureClassifier
from command_filter import CommandFilter

CONFIG_PATH = "config.yaml"
TRIALS_DIR = Path("data/trials")
REPORTS_DIR = Path("reports")
RAW_PREDS_CSV = REPORTS_DIR / "phase8_raw_predictions.csv"
CLASSES = [0, 1, 2, 3, 4]


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scan_trial_videos(base_dir: Path) -> List[Tuple[Path, Path]]:
    """Descubre parejas de video (.mp4) y metadatos (.json)."""
    trials = []
    if not base_dir.exists():
        return trials

    for json_path in sorted(base_dir.rglob("*.json")):
        video_path = json_path.with_suffix(".mp4")
        if video_path.exists():
            trials.append((video_path, json_path))
    return trials


def compute_and_cache_predictions(
    trials: List[Tuple[Path, Path]],
    classifier: GestureClassifier,
    out_csv: Path,
) -> None:
    """Ejecuta GestureClassifier.predict() sobre cada cuadro y almacena predicciones crudas."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    total_videos = len(trials)
    print(f"\n[INFERENCIA OFFLINE] Procesando {total_videos} videos de prueba...")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "frame_idx", "t", "clase_real", "condicion", "persona",
            "pred_cruda", "confianza", "hand_present", "lat_preproc", "lat_cnn", "lat_total"
        ])

        for v_idx, (v_path, m_path) in enumerate(trials, 1):
            with open(m_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)

            c_real = meta["clase_real"]
            cond = meta["condicion"]
            pers = meta["persona"]

            cap = cv2.VideoCapture(str(v_path))
            frame_idx = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if fps > 0 else 30.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t_sec = frame_idx / fps
                inf = classifier.predict(frame)

                writer.writerow([
                    v_path.name,
                    frame_idx,
                    round(t_sec, 4),
                    c_real,
                    cond,
                    pers,
                    inf["clase"],
                    round(inf["confianza"], 4),
                    int(inf["hand_present"]),
                    round(inf["latencia_ms"]["preproc"], 2),
                    round(inf["latencia_ms"]["cnn"], 2),
                    round(inf["latencia_ms"]["total"], 2),
                ])
                frame_idx += 1

            cap.release()
            print(f" -> [{v_idx:02d}/{total_videos}] {v_path.name}: {frame_idx} cuadros computados.")

    print(f"[CACHE COMPLETA] Predicciones crudas consolidadas en: {out_csv.resolve()}\n")


def evaluate_filter_pass(raw_csv: Path, cfg: Dict[str, Any]) -> None:
    """Aplica la FSM de CommandFilter sobre las predicciones crudas y diagnóstica causas raíz."""
    filter_engine = CommandFilter(cfg)

    # Contadores globales y matriz de confusión
    cm = np.zeros((5, 5), dtype=int)
    latencies = []
    condition_stats: Dict[str, Dict[str, int]] = {}

    table_rows = []
    current_video = ""

    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video"]
            if vid != current_video:
                filter_engine.reset_to_idle()
                current_video = vid

            t = float(row["t"])
            c_real = int(row["clase_real"])
            pred = int(row["pred_cruda"])
            conf = float(row["confianza"])
            present = bool(int(row["hand_present"]))
            lat = float(row["lat_total"])
            cond = row["condicion"]
            pers = row["persona"]

            latencies.append(lat)
            cm[c_real, pred] += 1

            # Inicializar estadísticas por condición
            if cond not in condition_stats:
                condition_stats[cond] = {"total_frames": 0, "correct_raw": 0, "accepted_correct": 0, "accepted_false": 0}
            condition_stats[cond]["total_frames"] += 1

            if pred == c_real:
                condition_stats[cond]["correct_raw"] += 1

            # Actualizar máquina de estados
            filt_res = filter_engine.update(pred, conf, t, hand_present=present)
            accepted = filt_res["comando_aceptado"]
            reason = filt_res["motivo_rechazo"]

            # Regla formal de diagnóstico causal solicitada:
            # - clase incorrecta -> "datos/modelo"
            # - clase correcta y comando rechazado -> "filtro"
            # - sin presencia de mano -> "seguridad"
            diagnostico = "OK"
            if accepted is not None:
                if accepted == c_real:
                    condition_stats[cond]["accepted_correct"] += 1
                else:
                    condition_stats[cond]["accepted_false"] += 1
                    diagnostico = "datos/modelo"
            else:
                if not present:
                    diagnostico = "seguridad"
                elif pred != c_real:
                    diagnostico = "datos/modelo"
                elif pred == c_real:
                    diagnostico = "filtro"

            table_rows.append({
                "video": vid,
                "t": t,
                "persona": pers,
                "condicion": cond,
                "clase_real": c_real,
                "pred_cruda": pred,
                "confianza": conf,
                "comando_aceptado": accepted if accepted is not None else "",
                "motivo_rechazo": reason if reason else "",
                "latencia_ms": lat,
                "diagnostico_error": diagnostico,
            })

    # Guardar reports/phase8_tabla.csv
    table_csv = REPORTS_DIR / "phase8_tabla.csv"
    with open(table_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "video", "t", "persona", "condicion", "clase_real", "pred_cruda",
            "confianza", "comando_aceptado", "motivo_rechazo", "latencia_ms", "diagnostico_error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    # Imprimir tablas de reporte
    print("\n" + "=" * 75)
    print(" DESGLOSE DE ROBUSTEZ POR CONDICIÓN AMBIENTAL ADVERSA")
    print("=" * 75)
    print(f"{'Condición':<20} | {'Cuadros':<8} | {'CNN Acc (%)':<12} | {'Cmd Correctos':<14} | {'Cmd Falsos':<10}")
    print("-" * 75)
    for cnd, st in condition_stats.items():
        acc = (st["correct_raw"] / st["total_frames"]) * 100.0 if st["total_frames"] > 0 else 0.0
        print(f"{cnd:<20} | {st['total_frames']:<8} | {acc:>10.1f}% | {st['accepted_correct']:<14} | {st['accepted_false']:<10}")
    print("=" * 75)

    # Matriz de Confusión Cuadro a Cuadro
    print("\n--- MATRIZ DE CONFUSIÓN CUADRO A CUADRO EN VIVO ---")
    header = "Real \\ Pred | " + " | ".join([f"  C{i}  " for i in CLASSES])
    print(header)
    print("-" * len(header))
    for i in CLASSES:
        row_str = f"    C{i}     | " + " | ".join([f"{cm[i, j]:6d}" for j in CLASSES])
        print(row_str)
    print("-" * len(header))

    # Diagnóstico general
    total_diag = len(table_rows)
    err_modelo = sum(1 for r in table_rows if r["diagnostico_error"] == "datos/modelo")
    err_filtro = sum(1 for r in table_rows if r["diagnostico_error"] == "filtro")
    err_seg = sum(1 for r in table_rows if r["diagnostico_error"] == "seguridad")

    print(f"\nResumen Causal de Diagnóstico sobre {total_diag} cuadros:")
    print(f" -> Fallos por Datos / Modelo CNN : {err_modelo} ({err_modelo/total_diag*100:.1f}%)")
    print(f" -> Bloqueos por Filtro (Cooldown/Ventana): {err_filtro} ({err_filtro/total_diag*100:.1f}%)")
    print(f" -> Bloqueos por Ausencia de Mano : {err_seg} ({err_seg/total_diag*100:.1f}%)")
    print(f"\n[OK] Tabla completa guardada en: {table_csv.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecución y auditoría formal de la Fase 8.")
    parser.add_argument("--repetir-filtro", action="store_true", help="Repite solo el CommandFilter usando predicciones cacheadas.")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trials = scan_trial_videos(TRIALS_DIR)

    # Auditoría de cantidad mínima: Exigir al menos 20 videos por clase en total
    class_counts = {c: 0 for c in CLASSES}
    for _, m_path in trials:
        with open(m_path, "r", encoding="utf-8") as f:
            m = json.load(f)
            class_counts[m["clase_real"]] += 1

    print("\n" + "=" * 65)
    print(" AUDITORÍA DE BANCO DE PRUEBAS (DATA/TRIALS)")
    print("=" * 65)
    for c in CLASSES:
        print(f" Clase {c}: {class_counts[c]} videos registrados.")
        if class_counts[c] < 20:
            print(f"  [AVISO] Clase {c} tiene menos de 20 videos ({class_counts[c]}/20 requeridos).")
    print("=" * 65)

    if not args.repetir_filtro or not RAW_PREDS_CSV.exists():
        classifier = GestureClassifier(CONFIG_PATH)
        compute_and_cache_predictions(trials, classifier, RAW_PREDS_CSV)
    else:
        print("\n[MODO REPETIR-FILTRO] Omitiendo inferencia CNN. Reusando predicciones crudas cacheadas.")

    evaluate_filter_pass(RAW_PREDS_CSV, cfg)


if __name__ == "__main__":
    main()