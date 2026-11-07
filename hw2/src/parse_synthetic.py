"""Минимальный парсер для синтетических данных.

Создаёт DataFrame в формате договора:
patient_id, time_h, measurement_1, measurement_2, measurement_3
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_patients(
    n_patients: int = 100,
    seed: int = 11,
    times: list[float] | None = None,
    noise_sigma: float = 0.2,
) -> pd.DataFrame:
    """Генерация синтетических пациентов.

    Parameters
    ----------
    n_patients : int
        Количество пациентов.
    seed : int
        Seed для воспроизводимости.
    times : list[float] or None
        Временные точки для обучения. По умолчанию [1, 2, 3.5, 24, 48].
    noise_sigma : float
        Стандартное отклонение аддитивного шума.

    Returns
    -------
    pd.DataFrame
        DataFrame с полями patient_id, time_h, measurement_1, measurement_2, measurement_3.
    """
    if times is None:
        times = [1.0, 2.0, 3.5, 24.0, 48.0]

    rng = np.random.RandomState(seed)

    # Генерация параметров
    A = rng.uniform(5.0, 30.0, n_patients)
    lam = rng.uniform(0.005, 0.03, n_patients)

    rows = []
    for i in range(n_patients):
        patient_id = f"SYNTH_{seed}_{i:04d}"
        for t in times:
            # Истинное значение модели
            true_val = 0.2 + A[i] * np.exp(-lam[i] * t)
            # Три измерения с шумом
            m = [true_val + rng.normal(0, noise_sigma) for _ in range(3)]
            rows.append({
                "patient_id": patient_id,
                "time_h": t,
                "measurement_1": m[0],
                "measurement_2": m[1],
                "measurement_3": m[2],
            })

    return pd.DataFrame(rows)


def generate_ground_truth(
    n_patients: int = 100,
    seed: int = 11,
    times: list[float] | None = None,
) -> pd.DataFrame:
    """Генерация истинных значений (без шума) для сравнения прогнозов.

    Returns
    -------
    pd.DataFrame
        DataFrame с полями patient_id, time_h, dose_rate (истинное значение).
    """
    if times is None:
        times = [1.0, 2.0, 3.5, 24.0, 48.0, 72.0]

    rng = np.random.RandomState(seed)
    A = rng.uniform(5.0, 30.0, n_patients)
    lam = rng.uniform(0.005, 0.03, n_patients)

    rows = []
    for i in range(n_patients):
        patient_id = f"SYNTH_{seed}_{i:04d}"
        for t in times:
            true_val = 0.2 + A[i] * np.exp(-lam[i] * t)
            rows.append({
                "patient_id": patient_id,
                "time_h": t,
                "dose_rate": true_val,
            })

    return pd.DataFrame(rows)