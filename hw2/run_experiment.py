#!/usr/bin/env python3
"""Запуск эксперимента: одноэкспоненциальная модель vs базовая линия.

Режимы:
    dry    — проверка схемы синтетических данных
    smoke  — один seed, проверка RMSE
    sweep  — все 5 seed
    full   — полный расчёт с выводом по правилу METHOD.md §7

Запуск:
    python run_experiment.py dry
    python run_experiment.py smoke
    python run_experiment.py sweep
    python run_experiment.py full
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.exponential import dose_rate, fit_exponential
from src.parse_synthetic import generate_synthetic_patients, generate_ground_truth

# ── Константы ──────────────────────────────────────────────────────────

SEEDS = [11, 22, 33, 44, 55]
N_PATIENTS = 100
FIT_TIMES = [1.0, 2.0, 3.5, 24.0, 48.0]
PREDICT_TIME = 72.0
NOISE_SIGMA = 0.2
RESULTS_DIR = Path("results")


# ── Вспомогательные функции ────────────────────────────────────────────

def _mean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Вычислить средние измерения для каждой (patient_id, time_h)."""
    df = df.copy()
    df["mean"] = df[["measurement_1", "measurement_2", "measurement_3"]].mean(axis=1)
    return df[["patient_id", "time_h", "mean"]]


def _fit_patient(group: pd.DataFrame) -> dict:
    """Подгонка модели для одного пациента."""
    t = group["time_h"].values
    y = group["mean"].values
    try:
        result = fit_exponential(t, y)
        return {"A": result.A, "lam": result.lam, "half_life": result.half_life, "success": True}
    except Exception:
        return {"A": None, "lam": None, "half_life": None, "success": False}


def _baseline_prediction(group: pd.DataFrame) -> float:
    """Базовая линия: среднее измерения на 48 ч."""
    row = group[group["time_h"] == 48.0]
    if row.empty:
        return np.nan
    return row["mean"].values[0]


def _exp_prediction(group: pd.DataFrame, A: float, lam: float) -> float:
    """Прогноз экспоненциальной модели на 72 ч."""
    return dose_rate(PREDICT_TIME, A, lam)


def _compute_rmse(predictions: list[float], truths: list[float]) -> float:
    """RMSE между прогнозами и истинными значениями."""
    preds = np.array(predictions)
    trues = np.array(truths)
    return float(np.sqrt(np.mean((preds - trues) ** 2)))


# ── Режимы ─────────────────────────────────────────────────────────────

def run_dry() -> int:
    """Проверка схемы синтетических данных."""
    print("=== DRY: проверка схемы ===")
    df = generate_synthetic_patients(n_patients=5, seed=11)
    required = ["patient_id", "time_h", "measurement_1", "measurement_2", "measurement_3"]
    assert all(c in df.columns for c in required), "Отсутствуют столбцы"
    assert df.shape[0] == 5 * len(FIT_TIMES), f"Неверное число строк: {df.shape[0]}"
    print(f"  Схема OK: {df.shape[0]} строк, {len(required)} столбцов")
    print("ПРОЙДЕН")
    return 0


def run_smoke() -> int:
    """Один seed, проверка RMSE."""
    print("=== SMOKE: один seed ===")
    seed = SEEDS[0]
    df = generate_synthetic_patients(n_patients=10, seed=seed)
    gt = generate_ground_truth(n_patients=10, seed=seed, times=FIT_TIMES + [PREDICT_TIME])

    means = _mean_measurements(df)

    # Группировка по пациентам
    grouped = means.groupby("patient_id")
    gt_grouped = gt.groupby("patient_id")

    exp_preds = []
    base_preds = []
    truths = []

    for pid, grp in grouped:
        gt_row = gt_grouped.get_group(pid)
        truth_72 = gt_row[gt_row["time_h"] == PREDICT_TIME]["dose_rate"].values[0]
        truths.append(truth_72)

        # Экспоненциальная модель
        fit_result = _fit_patient(grp)
        if fit_result["success"]:
            exp_pred = _exp_prediction(grp, fit_result["A"], fit_result["lam"])
            exp_preds.append(exp_pred)
        else:
            exp_preds.append(np.nan)

        # Базовая линия
        base_pred = _baseline_prediction(grp)
        base_preds.append(base_pred)

    rmse_exp = _compute_rmse(exp_preds, truths)
    rmse_base = _compute_rmse(base_preds, truths)
    print(f"  RMSE_exp = {rmse_exp:.4f}")
    print(f"  RMSE_baseline = {rmse_base:.4f}")
    print(f"  Δ = {rmse_base - rmse_exp:.4f}")
    print("ПРОЙДЕН")
    return 0


def run_sweep() -> int:
    """Все 5 seed, сохранение результатов."""
    print("=== SWEEP: все seed ===")
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {}
    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        df = generate_synthetic_patients(n_patients=N_PATIENTS, seed=seed)
        gt = generate_ground_truth(n_patients=N_PATIENTS, seed=seed, times=FIT_TIMES + [PREDICT_TIME])

        means = _mean_measurements(df)
        grouped = means.groupby("patient_id")
        gt_grouped = gt.groupby("patient_id")

        exp_preds = []
        base_preds = []
        truths = []

        for pid, grp in grouped:
            gt_row = gt_grouped.get_group(pid)
            truth_72 = gt_row[gt_row["time_h"] == PREDICT_TIME]["dose_rate"].values[0]
            truths.append(truth_72)

            fit_result = _fit_patient(grp)
            if fit_result["success"]:
                exp_pred = _exp_prediction(grp, fit_result["A"], fit_result["lam"])
                exp_preds.append(exp_pred)
            else:
                exp_preds.append(np.nan)

            base_pred = _baseline_prediction(grp)
            base_preds.append(base_pred)

        rmse_exp = _compute_rmse(exp_preds, truths)
        rmse_base = _compute_rmse(base_preds, truths)
        delta = rmse_base - rmse_exp

        results[seed] = {
            "rmse_exp": rmse_exp,
            "rmse_baseline": rmse_base,
            "delta": delta,
            "n_success": sum(1 for p in exp_preds if not np.isnan(p)),
        }
        print(f"    RMSE_exp={rmse_exp:.4f}, RMSE_base={rmse_base:.4f}, Δ={delta:.4f}")

    # Сохранение результатов
    out_path = RESULTS_DIR / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Результаты сохранены: {out_path}")
    print("ПРОЙДЕН")
    return 0


def run_full() -> int:
    """Полный расчёт с выводом по правилу METHOD.md §7."""
    print("=== FULL: полный расчёт ===")
    RESULTS_DIR.mkdir(exist_ok=True)

    deltas = []
    all_results = {}

    for seed in SEEDS:
        print(f"\n  Seed {seed}...")
        df = generate_synthetic_patients(n_patients=N_PATIENTS, seed=seed)
        gt = generate_ground_truth(n_patients=N_PATIENTS, seed=seed, times=FIT_TIMES + [PREDICT_TIME])

        means = _mean_measurements(df)
        grouped = means.groupby("patient_id")
        gt_grouped = gt.groupby("patient_id")

        exp_preds = []
        base_preds = []
        truths = []

        for pid, grp in grouped:
            gt_row = gt_grouped.get_group(pid)
            truth_72 = gt_row[gt_row["time_h"] == PREDICT_TIME]["dose_rate"].values[0]
            truths.append(truth_72)

            fit_result = _fit_patient(grp)
            if fit_result["success"]:
                exp_pred = _exp_prediction(grp, fit_result["A"], fit_result["lam"])
                exp_preds.append(exp_pred)
            else:
                exp_preds.append(np.nan)

            base_pred = _baseline_prediction(grp)
            base_preds.append(base_pred)

        rmse_exp = _compute_rmse(exp_preds, truths)
        rmse_base = _compute_rmse(base_preds, truths)
        delta = rmse_base - rmse_exp
        deltas.append(delta)

        all_results[seed] = {
            "rmse_exp": rmse_exp,
            "rmse_baseline": rmse_base,
            "delta": delta,
        }
        print(f"    RMSE_exp={rmse_exp:.4f}, RMSE_base={rmse_base:.4f}, Δ={delta:.4f}")

    # Парное сравнение
    deltas = np.array(deltas)
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1))
    n = len(deltas)
    df_t = n - 1
    ci_lower, ci_upper = stats.t.interval(0.95, df=df_t, loc=mean_delta, scale=std_delta / np.sqrt(n))

    print(f"\n=== ПАРАЯ РАЗНОСТЬ ===")
    print(f"  Среднее Δ = {mean_delta:.4f}")
    print(f"  Стд. откл = {std_delta:.4f}")
    print(f"  95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Практический порог
    practical_threshold = 0.2
    practical_ok = mean_delta > practical_threshold
    print(f"  Практический порог ({practical_threshold}): {'ПРОЙДЕН' if practical_ok else 'НЕ ПРОЙДЕН'}")

    # Статистическая значимость
    stat_sig = ci_lower > 0
    print(f"  Стат. значимость (нижняя граница > 0): {'ДА' if stat_sig else 'НЕТ'}")

    # Итоговый вердикт
    if stat_sig and practical_ok:
        verdict = "преимущество подтверждено"
    elif stat_sig:
        verdict = "статистически значимо, но практический порог не достигнут"
    else:
        verdict = "преимущество над базовой линией не подтверждено"

    print(f"\n  ИТОГОВЫЙ ВЕРДИКТ: {verdict}")

    # Сохранение
    out = {
        "per_seed": all_results,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_95": [ci_lower, ci_upper],
        "verdict": verdict,
    }
    out_path = RESULTS_DIR / "full_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Результаты сохранены: {out_path}")
    print("ПРОЙДЕН")
    return 0


# ── Точка входа ────────────────────────────────────────────────────────

MODES = {
    "dry": run_dry,
    "smoke": run_smoke,
    "sweep": run_sweep,
    "full": run_full,
}


def main() -> int:
    if len(sys.argv) < 2:
        print(f"Использование: python run_experiment.py {'|'.join(MODES)}")
        return 1

    mode = sys.argv[1]
    if mode not in MODES:
        print(f"Неизвестный режим: {mode}. Доступны: {', '.join(MODES)}")
        return 1

    return MODES[mode]()


if __name__ == "__main__":
    sys.exit(main())