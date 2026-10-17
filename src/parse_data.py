"""
Парсинг данных из Excel-файла пациентов РНТ I-131.
Извлекает временные точки и значения МАЭД(3) для каждого пациента.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
EXCEL_FILE = DATA_DIR / "Пациенты РНТ I-131.xlsx"

# Столбцы, содержащие данные МАЭД(3) — строковые, с тремя измерениями
MAED3_COLS = [
    "МАЭД(3) №1 мкЗв/час",
    "МАЭД(3) №2, мкЗв/час",
    "МАЭД(3) №3, мкЗв/час",
    "МАЭД(3) №4, мкЗв/час",
    "МАЭД(3) №5, мкЗв/час",
    "МАЭД(3) №6, мкЗв/час",
    "МАЭД(3) №7, мкЗв/час",
]

# Столбец с временными точками
TIME_COL = "Время измерения МАЭД после введения, час"


def parse_time_points(time_str: str) -> list[float]:
    """Парсит строку временных точек: '1 2 3,5 27 47 68' -> [1.0, 2.0, 3.5, 27.0, 47.0, 68.0]"""
    time_str = time_str.strip()
    # Заменяем запятую на точку (европейский формат)
    time_str = time_str.replace(",", ".")
    parts = time_str.split()
    return [float(p) for p in parts]


def parse_maed3_values(val_str: str) -> list[float] | None:
    """Парсит строку трёх измерений: '25,11 24,75 24,93' -> [25.11, 24.75, 24.93]"""
    if pd.isna(val_str) or val_str.strip() == "":
        return None
    val_str = val_str.strip()
    # Заменяем запятую на точку
    val_str = val_str.replace(",", ".")
    parts = val_str.split()
    if len(parts) != 3:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def parse_patient_data(row: pd.Series) -> dict | None:
    """
    Извлекает данные пациента из строки DataFrame.
    Возвращает dict с patient_id, times, measurements или None если данные неполные.
    """
    patient_id = row["№"]
    
    # Временные точки
    time_str = row.get(TIME_COL)
    if pd.isna(time_str) or str(time_str).strip() == "":
        return None
    
    times = parse_time_points(str(time_str))
    n_points = len(times)
    
    if n_points == 0:
        return None
    
    # Извлекаем измерения МАЭД(3) — берём первые n_points столбцов
    measurements = []
    valid = True
    
    for i in range(n_points):
        col_name = MAED3_COLS[i] if i < len(MAED3_COLS) else None
        if col_name is None:
            valid = False
            break
        
        val = row.get(col_name)
        vals = parse_maed3_values(str(val)) if not pd.isna(val) else None
        
        if vals is None:
            valid = False
            break
        
        measurements.append(vals)
    
    if not valid:
        return None
    
    # Проверяем что все измерения — по 3 значения
    for m in measurements:
        if len(m) != 3:
            return None
    
    # Средние значения для каждой временной точки
    means = [np.mean(m) for m in measurements]
    
    return {
        "patient_id": int(patient_id),
        "times": times,
        "measurements": measurements,  # list of list[float] — по 3 на точку
        "means": means,
        "n_points": n_points,
    }


def process_sheet(sheet_name: str, df: pd.DataFrame) -> list[dict]:
    """Обрабатывает один лист Excel."""
    results = []
    
    for _, row in df.iterrows():
        data = parse_patient_data(row)
        if data is not None:
            data["sheet"] = sheet_name
            results.append(data)
    
    return results


def main():
    xl = pd.ExcelFile(EXCEL_FILE)
    all_patients = []
    skipped = []
    
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet)
        print(f"\n=== Лист: {sheet} ===")
        print(f"  Строк: {len(df)}")
        
        # Проверяем наличие нужных столбцов
        has_time = TIME_COL in df.columns
        has_maed3 = any(c in df.columns for c in MAED3_COLS)
        
        if not has_time or not has_maed3:
            print(f"  ПРОПУСК: нет нужных столбцов (time={has_time}, maed3={has_maed3})")
            skipped.append((sheet, "нет нужных столбцов"))
            continue
        
        sheet_patients = process_sheet(sheet, df)
        print(f"  Валидных пациентов: {len(sheet_patients)}")
        all_patients.extend(sheet_patients)
        
        # Статистика по количеству точек
        n_points_dist = {}
        for p in sheet_patients:
            np_ = p["n_points"]
            n_points_dist[np_] = n_points_dist.get(np_, 0) + 1
        print(f"  Распределение по точкам: {n_points_dist}")
    
    print(f"\n=== ИТОГО ===")
    print(f"  Валидных пациентов: {len(all_patients)}")
    print(f"  Пропущено листов: {len(skipped)}")
    
    # Проверка примера из data_example.txt
    # Пациент №163: Временные точки: 1 2 3,5 27 47 68
    # Пациент №172: Временные точки: 1 2 3 24 48 68
    example_163 = [p for p in all_patients if p["patient_id"] == 163]
    example_172 = [p for p in all_patients if p["patient_id"] == 172]
    
    if example_163:
        p = example_163[0]
        print(f"\nПроверка пациента 163:")
        print(f"  times: {p['times']}")
        print(f"  means: {p['means']}")
        print(f"  measurements: {p['measurements']}")
    else:
        print("\nПациент 163 НЕ НАЙДЕН")
    
    if example_172:
        p = example_172[0]
        print(f"\nПроверка пациента 172:")
        print(f"  times: {p['times']}")
        print(f"  means: {p['means']}")
    else:
        print("Пациент 172 НЕ НАЙДЕН")
    
    # Сохраняем распарсенные данные
    import json
    output = []
    for p in all_patients:
        output.append({
            "patient_id": p["patient_id"],
            "times": p["times"],
            "means": p["means"],
            "n_points": p["n_points"],
            "sheet": p["sheet"],
        })
    
    out_path = Path(__file__).parent.parent / "result" / "parsed_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено в {out_path}")
    
    return all_patients


if __name__ == "__main__":
    main()