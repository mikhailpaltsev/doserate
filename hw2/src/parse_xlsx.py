"""Парсер XLSX для данных измерений МАЭД после I-131.

Алгоритм основан на prepare_data.ipynb:
- Чтение всех листов, поиск столбцов по заголовкам
- Парсинг временных точек и тройных измерений
- Проверка целостности (число времён = число троек, ровно 3 измерения)
- Нормализация в DataFrame: patient_id, time_h, measurement_1/2/3
- Сохранение реестра отклонённых записей

Для синтетических данных используется упрощённый формат CSV/XLSX.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any
import numpy as np

import pandas as pd


def _is_empty(x: Any) -> bool:
    """Проверка на пустое значение."""
    return pd.isna(x) or str(x).strip() == ""


def _parse_numbers(x: Any) -> list[float]:
    """Парсинг строки чисел: '1 2 3,5' -> [1.0, 2.0, 3.5]."""
    return [
        float(v.replace(",", "."))
        for v in str(x).split()
    ]


def _patient_key(x: Any) -> str:
    """Номер пациента из первого столбца."""
    n = float(str(x).replace(",", "."))
    if not n.is_integer():
        raise ValueError("Номер пациента не целый")
    return str(int(n))


def parse_xlsx(
    path: str | Path,
    id_mapping: dict[tuple[str, str], str] | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    """Чтение XLSX и преобразование в нормализованный DataFrame.

    Parameters
    ----------
    path : str or Path
        Путь к файлу XLSX.
    id_mapping : dict or None
        Карта соответствия исходных номеров пациентов псевдонимам.
        Если None, создаётся новая.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        (измерения, отклонённые записи).
    """
    if id_mapping is None:
        id_mapping = {}

    sheets = pd.read_excel(path, sheet_name=None, dtype=object)

    records: list[dict] = []
    rejected: list[dict] = []

    for sheet_name, df in sheets.items():
        headers = {
            str(c).strip().replace("\xa0", " "): c
            for c in df.columns
        }

        id_col = next(
            (v for k, v in headers.items()
             if k in ("№", "№ пациента")), None
        )

        time_col = next(
            (v for k, v in headers.items()
             if "Время измерения МАЭД после введения" in k),
            None
        )

        dose_cols: dict[int, Any] = {}
        for header, col in headers.items():
            match = re.search(
                r"МАЭД\s*\(\s*3\s*\)\s*№\s*(\d+)",
                header, re.IGNORECASE
            )
            if match:
                dose_cols[int(match.group(1))] = col

        if id_col is None or time_col is None or not dose_cols:
            continue

        for idx, row in df.iterrows():
            if _is_empty(row[id_col]):
                continue

            try:
                source_id = _patient_key(row[id_col])
            except (ValueError, TypeError):
                continue

            if source_id not in id_mapping:
                id_mapping[source_id] = f"P{len(id_mapping) + 1:04d}"
            pid = id_mapping[source_id]

            try:
                if _is_empty(row[time_col]):
                    raise ValueError("Нет временных точек")

                times = _parse_numbers(row[time_col])

                measurements = {
                    n: _parse_numbers(row[col])
                    for n, col in sorted(dose_cols.items())
                    if not _is_empty(row[col])
                }

                if len(times) != len(measurements):
                    raise ValueError(
                        f"Времён: {len(times)}, троек: {len(measurements)}"
                    )

                if list(measurements) != list(range(1, len(times) + 1)):
                    raise ValueError("Пропущены или смещены номера измерений")

                if any(len(m) != 3 for m in measurements.values()):
                    raise ValueError("Не ровно 3 измерения на точку")

                if not all(math.isfinite(t) and t >= 0 for t in times):
                    raise ValueError("Некорректное время")

                if len(set(times)) != len(times):
                    raise ValueError("Повторяющиеся временные точки")

                if not all(
                    math.isfinite(v) and v >= 0
                    for m in measurements.values() for v in m
                ):
                    raise ValueError("Некорректная мощность дозы")

                for n, t in enumerate(times, start=1):
                    a, b, c = measurements[n]
                    records.append({
                        "patient_id": pid,
                        "time_h": t,
                        "measurement_1": a,
                        "measurement_2": b,
                        "measurement_3": c
                    })

            except (ValueError, TypeError) as e:
                rejected.append({
                    "patient_id": pid,
                    "sheet": sheet_name,
                    "excel_row": idx + 2,
                    "reason": str(e)
                })

    columns = [
        "patient_id", "time_h",
        "measurement_1", "measurement_2", "measurement_3"
    ]
    result = pd.DataFrame(records, columns=columns)
    result = result.sort_values(["patient_id", "time_h"])

    return result, rejected


def generate_synthetic_xlsx(
    path: str | Path,
    n_patients: int = 100,
    seed: int = 11,
    times: list[float] | None = None,
    noise_sigma: float = 0.2,
) -> None:
    """Генерация синтетического XLSX для тестирования парсера.

    Создаёт файл с одним листом 'Sheet1' в формате, совместимом с
    реальным XLSX (столбцы: №, Время измерения МАЭД..., МАЭД(3) №1-3).
    """
    if times is None:
        times = [1.0, 2.0, 3.5, 24.0, 48.0]

    rng = __import__("numpy").random.RandomState(seed)

    rows = []
    for i in range(n_patients):
        pid = f"{i + 1:04d}"
        A = rng.uniform(5.0, 30.0)
        lam = rng.uniform(0.005, 0.03)
        time_str_parts = []
        dose_parts = {}

        for n, t in enumerate(times, start=1):
            true_val = 0.2 + A * np.exp(-lam * t)

            time_str_parts.append(str(t))

            dose_parts[n] = [
                float(true_val + rng.normal(0, noise_sigma))
                for _ in range(3)]

        row = {
            "№": pid,
            "Время измерения МАЭД после введения I-131 (ч)": " ".join(time_str_parts),
        }
        for n in range(1, len(times) + 1):
            row[f"МАЭД(3) №{n}"] = " ".join(str(v) for v in dose_parts[n])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, engine="openpyxl")