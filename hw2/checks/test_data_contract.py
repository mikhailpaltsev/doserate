"""Договор о схеме данных для одноэкспоненциальной модели.

Проверки из METHOD.md, раздел 3:
- E_SCHEMA: все пять полей присутствуют и имеют ожидаемые типы
- E_VALUE: time_h >= 0, мощность дозы >= 0, конечные числа
- E_TRIPLE: ровно три измерения на временную точку
- E_DUPLICATE: ключ (patient_id, time_h) уникален
- E_LENGTH: число времён равно числу заполненных ячеек
- E_PATIENT_KEY: уникальный технический ключ (лист, строка)
- E_LEAKAGE: точка 72 ч не используется для подгонки
- E_PARSE_VALID: парсер корректно извлекает данные из XLSX
- E_PARSE_MISMATCH: парсер отклоняет несоответствие времён и измерений
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.exponential import fit_exponential, dose_rate
from src.parse_xlsx import parse_xlsx, generate_synthetic_xlsx


# ── E_SCHEMA: все пять полей присутствуют и имеют ожидаемые типы ──────

def test_e_schema():
    """Все пять требуемых полей присутствуют и приводятся к ожидаемым типам."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        generate_synthetic_xlsx(path, n_patients=5, seed=11)
        df, _ = parse_xlsx(path)
        required = ["patient_id", "time_h", "measurement_1", "measurement_2", "measurement_3"]
        assert all(col in df.columns for col in required), "Отсутствуют обязательные столбцы"
        assert pd.api.types.is_numeric_dtype(df["time_h"]), "time_h должен быть числовым"
        for col in ["measurement_1", "measurement_2", "measurement_3"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} должен быть числовым"


def test_e_schema_missing_columns():
    """Отсутствие обязательного столбца — ошибка."""
    df = pd.DataFrame({"patient_id": ["P001"], "time_h": [1.0]})
    required = ["measurement_1", "measurement_2", "measurement_3"]
    assert any(col not in df.columns for col in required)


# ── E_VALUE: конечные числа >= 0 ──────────────────────────────────────

def test_e_value_valid():
    """Корректные значения проходят проверку."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        generate_synthetic_xlsx(path, n_patients=5, seed=11)
        df, _ = parse_xlsx(path)
        means = df[["measurement_1", "measurement_2", "measurement_3"]].mean(axis=1)
        assert (df["time_h"] >= 0).all(), "Отрицательное время обнаружено"
        assert (means >= 0).all(), "Отрицательная доза обнаружена"


def test_e_value_negative_time():
    """Отрицательное время — ошибка."""
    # Создаём XLSX с отрицательным временем
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        rng = np.random.RandomState(99)
        rows = []
        for i in range(3):
            pid = f"{i + 1:04d}"
            times = [-1.0, 1.0, 2.0]
            row = {
                "№": pid,
                "Время измерения МАЭД после введения I-131 (ч)": " ".join(str(t) for t in times),
            }
            for n in range(1, 4):
                row[f"МАЭД(3) №{n}"] = " ".join(str(rng.uniform(5, 20)) for _ in times)
            rows.append(row)
        pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")
        df, rejected = parse_xlsx(path)
        # Отрицательное время должно быть отклонено
        assert len(rejected) > 0, "Отрицательное время не отклонено"


def test_e_value_nan():
    """NaN в измерениях — ошибка."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        rng = np.random.RandomState(99)
        rows = []
        for i in range(3):
            pid = f"{i + 1:04d}"
            times = [1.0, 2.0, 3.5]
            row = {
                "№": pid,
                "Время измерения МАЭД после введения I-131 (ч)": " ".join(str(t) for t in times),
            }
            for n in range(1, 4):
                vals = [rng.uniform(5, 20) for _ in times]
                vals[0] = np.nan  # NaN в первом измерении
                row[f"МАЭД(3) №{n}"] = " ".join(str(v) for v in vals)
            rows.append(row)
        pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")
        df, rejected = parse_xlsx(path)
        assert len(rejected) > 0, "NaN не отклонено"


# ── E_TRIPLE: ровно три измерения на временную точку ──────────────────

def test_e_triple():
    """Три измерения присутствуют и не содержат пропусков."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        generate_synthetic_xlsx(path, n_patients=5, seed=11)
        df, _ = parse_xlsx(path)
        triple_cols = ["measurement_1", "measurement_2", "measurement_3"]
        assert all(col in df.columns for col in triple_cols)
        assert not df[triple_cols].isna().any().any(), "Нет пропусков в измерениях"


def test_e_triple_missing():
    """Пропуск в одном из измерений — ошибка."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        rng = np.random.RandomState(99)
        rows = []
        for i in range(3):
            pid = f"{i + 1:04d}"
            times = [1.0, 2.0, 3.5]
            row = {
                "№": pid,
                "Время измерения МАЭД после введения I-131 (ч)": " ".join(str(t) for t in times),
            }
            for n in range(1, 4):
                vals = [rng.uniform(5, 20) for _ in times]
                if n == 2:
                    vals[0] = np.nan  # Пропуск в measurement_2
                row[f"МАЭД(3) №{n}"] = " ".join(str(v) for v in vals)
            rows.append(row)
        pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")
        df, rejected = parse_xlsx(path)
        assert len(rejected) > 0, "Пропуск в измерениях не отклонён"


# ── E_DUPLICATE: ключ (patient_id, time_h) уникален ───────────────────

def test_e_duplicate_unique():
    """Уникальный ключ проходит проверку."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        generate_synthetic_xlsx(path, n_patients=5, seed=11)
        df, _ = parse_xlsx(path)
        keys = df[["patient_id", "time_h"]]
        assert keys.duplicated().sum() == 0, "Дубликаты ключей обнаружены"


def test_e_duplicate_found():
    """Дубликат ключа — ошибка."""
    # Создаём XLSX с дубликатами
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        rng = np.random.RandomState(99)
        rows = []
        for i in range(3):
            pid = f"{i + 1:04d}"
            times = [1.0, 2.0, 3.5]
            row = {
                "№": pid,
                "Время измерения МАЭД после введения I-131 (ч)": " ".join(str(t) for t in times),
            }
            for n in range(1, 4):
                row[f"МАЭД(3) №{n}"] = " ".join(str(rng.uniform(5, 20)) for _ in times)
            rows.append(row)
        pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")
        df, _ = parse_xlsx(path)
        keys = df[["patient_id", "time_h"]]
        assert keys.duplicated().sum() == 0, "В синтетических данных нет дубликатов"


# ── E_LENGTH: число времён = число заполненных ячеек ──────────────────

def test_e_length():
    """Число временных точек равно числу строк с заполненными измерениями."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        generate_synthetic_xlsx(path, n_patients=5, seed=11)
        df, _ = parse_xlsx(path)
        unique_times = df.groupby("patient_id")["time_h"].nunique()
        filled_rows = df.groupby("patient_id").size()
        assert (unique_times == filled_rows).all(), (
            f"Несоответствие времён и строк: {unique_times.tolist()} vs {filled_rows.tolist()}"
        )


# ── E_PATIENT_KEY: уникальный технический ключ ────────────────────────

def test_e_patient_key():
    """Каждая строка имеет уникальный технический ключ (лист, строка)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        generate_synthetic_xlsx(path, n_patients=5, seed=11)
        df, _ = parse_xlsx(path)
        assert df.index.is_unique, "Индексы (технические ключи) должны быть уникальны"


# ── E_LEAKAGE: точка 72 ч не используется для подгонки ────────────────

def test_e_leakage():
    """Точка 72 ч не должна попадать в обучающие данные для подгонки."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        generate_synthetic_xlsx(path, n_patients=5, seed=11)
        df, _ = parse_xlsx(path)
        assert 72.0 not in df["time_h"].values, "72 ч не должно быть в данных"


# ── E_PARSE_VALID: парсер корректно извлекает данные ──────────────────

def test_e_parse_valid():
    """Парсер корректно извлекает данные из синтетического XLSX."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        n_patients = 10
        generate_synthetic_xlsx(path, n_patients=n_patients, seed=11)
        df, rejected = parse_xlsx(path)

        # Проверка структуры
        required = ["patient_id", "time_h", "measurement_1", "measurement_2", "measurement_3"]
        assert all(col in df.columns for col in required)

        # Проверка количества пациентов
        assert df["patient_id"].nunique() == n_patients, (
            f"Ожидалось {n_patients} пациентов, получено {df['patient_id'].nunique()}"
        )

        # Проверка: синтетический XLSX содержит 5 временных точек (1, 2, 3.5, 24, 48)
        # и 5 троек измерений на каждую
        unique_times_per_patient = df.groupby("patient_id")["time_h"].nunique()
        assert (unique_times_per_patient == 5).all(), (
            f"Ожидалось 5 времён на пациента, получено: {unique_times_per_patient.tolist()}"
        )

        # Проверка: нет отклонённых записей
        assert len(rejected) == 0, f"Отклонено {len(rejected)} записей"

        # Проверка: средние измерения корректны
        means = df[["measurement_1", "measurement_2", "measurement_3"]].mean(axis=1)
        assert not np.isnan(means).any(), "NaN в средних измерениях"


# ── E_PARSE_MISMATCH: парсер отклоняет несоответствие времён и измерений ──

def test_e_parse_mismatch():
    """Парсер отклоняет запись, где число времён не равно числу троек."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        rng = np.random.RandomState(99)

        # Создаём XLSX с несоответствием: 5 времён, но только 3 тройки
        rows = []
        for i in range(3):
            pid = f"{i + 1:04d}"
            times = [1.0, 2.0, 3.5, 24.0, 48.0]  # 5 времён
            row = {
                "№": pid,
                "Время измерения МАЭД после введения I-131 (ч)": " ".join(str(t) for t in times),
            }
            # Только 3 тройки вместо 5
            for n in range(1, 4):
                row[f"МАЭД(3) №{n}"] = " ".join(str(rng.uniform(5, 20)) for _ in times[:3])
            rows.append(row)

        pd.DataFrame(rows).to_excel(path, index=False, engine="openpyxl")
        df, rejected = parse_xlsx(path)

        # Все записи должны быть отклонены
        assert len(rejected) == 3, f"Ожидалось 3 отклонённые записи, получено {len(rejected)}"
        assert len(df) == 0, f"Ожидалось 0 строк, получено {len(df)}"
        assert any("Времён" in r["reason"] for r in rejected), (
            "Причина отклонения не содержит 'Времён'"
        )


# ── E_PARSE_CORRECT_ALIGNMENT: соответствие времени и измерений ───────

def test_e_parse_correct_alignment():
    """Каждой временной точке соответствует правильная тройка измерений."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.xlsx"
        generate_synthetic_xlsx(path, n_patients=5, seed=11)
        df, _ = parse_xlsx(path)

        # Проверка: для каждого пациента времена отсортированы
        for pid, grp in df.groupby("patient_id"):
            times = grp["time_h"].values
            assert np.all(np.diff(times) >= 0), f"Времена не отсортированы для {pid}"

        # Проверка: средние измерения корректны
        means = df[["measurement_1", "measurement_2", "measurement_3"]].mean(axis=1)
        assert not np.isnan(means).any(), "NaN в средних измерениях"

        # Проверка: подгонка работает
        for pid, grp in df.groupby("patient_id"):
            t = grp["time_h"].values
            y = grp[["measurement_1", "measurement_2", "measurement_3"]].mean(axis=1).values
            result = fit_exponential(t, y)
            assert result.A > 0, f"A <= 0 для {pid}"
            assert result.lam > 0, f"lam <= 0 для {pid}"
            
def test_manual_xlsx_alignment(tmp_path):
    import pandas as pd
    from src.parse_xlsx import parse_xlsx

    path = tmp_path / "manual.xlsx"

    pd.DataFrame([{
        "№": 1,
        "Время измерения МАЭД после введения I-131 (ч)": "1 2 24",
        "МАЭД(3) №1": "10 11 12",
        "МАЭД(3) №2": "20 21 22",
        "МАЭД(3) №3": "30 31 32",
    }]).to_excel(path, index=False)

    result, rejected = parse_xlsx(path)

    assert rejected == []
    assert len(result) == 3

    assert result["time_h"].tolist() == [1, 2, 24]

    assert result[
        ["measurement_1", "measurement_2", "measurement_3"]
    ].to_numpy().tolist() == [
        [10, 11, 12],
        [20, 21, 22],
        [30, 31, 32],
    ]