"""
Подбор 3 лучших моделей экспоненциального распада и расчёт T1/2.
Модели:
  M1: однокомпонентная  A(t) = A0 * exp(-λ*t)
  M2: двухкомпонентная  A(t) = A1*exp(-λ1*t) + A2*exp(-λ2*t)
  M3: трёхкомпонентная  A(t) = A1*exp(-λ1*t) + A2*exp(-λ2*t) + A3*exp(-λ3*t)

Метрики: R², RMSE, AIC.
T1/2: для M1 — ln(2)/λ; для M2 — эффективный T1/2; для M3 — эффективный T1/2.
"""

import json
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path

RESULT_DIR = Path(__file__).parent.parent / "result"
PARSED_FILE = RESULT_DIR / "parsed_data.json"


# --- Модели ---

def model_single(t, A0, lam):
    """Однокомпонентная экспонента."""
    return A0 * np.exp(-lam * t)


def model_double(t, A1, lam1, A2, lam2):
    """Двухкомпонентная экспонента."""
    return A1 * np.exp(-lam1 * t) + A2 * np.exp(-lam2 * t)


def model_triple(t, A1, lam1, A2, lam2, A3, lam3):
    """Трёхкомпонентная экспонента."""
    return (A1 * np.exp(-lam1 * t) 
            + A2 * np.exp(-lam2 * t) 
            + A3 * np.exp(-lam3 * t))


MODELS = [
    ("M1_single", model_single, 2, "A0, lam"),
    ("M2_double", model_double, 4, "A1, lam1, A2, lam2"),
    ("M3_triple", model_triple, 6, "A1, lam1, A2, lam2, A3, lam3"),
]


# --- Метрики ---

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def aic(n, n_params, residuals):
    """AIC = n*ln(RSS/n) + 2*k"""
    rss = np.sum(residuals ** 2)
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + 2 * n_params


# --- Расчёт T1/2 ---

def effective_half_life(t, means, params, model_func):
    """
    Эффективный T1/2 — время, за которое активность падает до половины
    от начального значения (среднее от первой и последней точки).
    """
    A_start = means[0]
    A_target = A_start / 2
    
    # Ищем время когда модель достигает половины
    popt, _ = curve_fit(model_func, t, means, p0=params, maxfev=10000)
    
    # Вычисляем модель на плотной сетке
    t_fine = np.linspace(t[0], t[-1], 1000)
    y_fine = model_func(t_fine, *popt)
    
    # Ищем пересечение с A_target
    for i in range(1, len(y_fine)):
        if y_fine[i] <= A_target and y_fine[i-1] > A_target:
            # Линейная интерполяция
            frac = (A_target - y_fine[i-1]) / (y_fine[i] - y_fine[i-1])
            return float(t_fine[i-1] + frac * (t_fine[i] - t_fine[i-1]))
    
    # Если не достигли половины — возвращаем последнее время
    return float(t[-1])


def calc_t12_single(t, means):
    """T1/2 для однокомпонентной модели."""
    try:
        popt, _ = curve_fit(model_single, t, means, p0=[means[0], 0.1], maxfev=10000)
        lam = popt[1]
        if lam > 0:
            return float(np.log(2) / lam)
    except Exception:
        pass
    return None


def calc_t12_effective(t, means, model_func, n_params):
    """Эффективный T1/2 для многокомпонентной модели."""
    try:
        # Начальные параметры
        if n_params == 4:
            p0 = [means[0] * 0.7, 0.1, means[0] * 0.3, 0.01]
        else:  # 6
            p0 = [means[0] * 0.5, 0.2, means[0] * 0.3, 0.05, means[0] * 0.2, 0.005]
        
        popt, _ = curve_fit(model_func, t, means, p0=p0, maxfev=10000)
        
        # Проверяем что все λ > 0
        for i in range(1, n_params, 2):
            if popt[i] <= 0:
                return None
        
        return effective_half_life(t, means, popt, model_func)
    except Exception:
        return None


# --- Основной цикл ---

def fit_patient(patient: dict) -> dict:
    """Фитит все 3 модели для одного пациента."""
    t = np.array(patient["times"])
    means = np.array(patient["means"])
    n = len(t)
    
    results = {}
    
    for name, model_func, n_params, _ in MODELS:
        try:
            # Подбираем параметры
            if name == "M1_single":
                p0 = [means[0], 0.1]
            elif name == "M2_double":
                p0 = [means[0] * 0.7, 0.1, means[0] * 0.3, 0.01]
            else:
                p0 = [means[0] * 0.5, 0.2, means[0] * 0.3, 0.05, means[0] * 0.2, 0.005]
            
            popt, pcov = curve_fit(model_func, t, means, p0=p0, maxfev=10000)
            
            y_pred = model_func(t, *popt)
            residuals = means - y_pred
            
            r2 = r_squared(means, y_pred)
            rmse_val = rmse(means, y_pred)
            aic_val = aic(n, n_params, residuals)
            
            # T1/2
            if name == "M1_single":
                t12 = calc_t12_single(t, means)
            else:
                t12 = calc_t12_effective(t, means, model_func, n_params)
            
            results[name] = {
                "params": popt.tolist(),
                "R2": round(r2, 6),
                "RMSE": round(rmse_val, 6),
                "AIC": round(aic_val, 4),
                "T1/2": round(t12, 4) if t12 is not None else None,
            }
        except Exception as e:
            results[name] = {
                "params": None,
                "R2": None,
                "RMSE": None,
                "AIC": None,
                "T1/2": None,
                "error": str(e),
            }
    
    return results


def main():
    with open(PARSED_FILE) as f:
        patients = json.load(f)
    
    print(f"Обработка {len(patients)} пациентов...")
    
    all_results = []
    skipped = 0
    
    for i, patient in enumerate(patients):
        pid = patient["patient_id"]
        fit = fit_patient(patient)
        
        # Проверяем что хотя бы одна модель сошлась
        has_valid = any(
            fit[m]["R2"] is not None for m in ["M1_single", "M2_double", "M3_triple"]
        )
        
        if not has_valid:
            skipped += 1
            continue
        
        # Выбираем 3 лучшие модели по AIC (из доступных)
        valid_models = []
        for m in ["M1_single", "M2_double", "M3_triple"]:
            if fit[m]["R2"] is not None:
                valid_models.append((m, fit[m]["AIC"], fit[m]["T1/2"]))
        
        valid_models.sort(key=lambda x: x[1])  # по AIC (меньше = лучше)
        top3 = valid_models[:3]
        
        # Берём T1/2 из лучшей модели
        best_t12 = top3[0][2] if top3[0][2] is not None else None
        
        row = {
            "patient_id": pid,
        }
        
        # Добавляем T1/2 и метрики для топ-3 моделей
        for rank, (mname, aic_val, t12_val) in enumerate(top3, 1):
            row[f"T1/2_model_{rank}"] = round(t12_val, 4) if t12_val is not None else None
            row[f"R2_model_{rank}"] = fit[mname]["R2"]
            row[f"RMSE_model_{rank}"] = fit[mname]["RMSE"]
            row[f"AIC_model_{rank}"] = round(aic_val, 4)
            row[f"best_model_{rank}"] = mname
        
        # T1/2 из лучшей модели (для основного отчёта)
        row["T1/2_best"] = round(best_t12, 4) if best_t12 is not None else None
        
        all_results.append(row)
    
    print(f"Обработано: {len(all_results)}, пропущено: {skipped}")
    
    # Сохраняем результат
    out_path = RESULT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены в {out_path}")
    
    # Статистика по моделям
    model_counts = {"M1_single": 0, "M2_double": 0, "M3_triple": 0}
    for r in all_results:
        for k in ["best_model_1", "best_model_2", "best_model_3"]:
            m = r.get(k)
            if m:
                model_counts[m] = model_counts.get(m, 0) + 1
    
    print(f"\nРаспределение лучших моделей:")
    for m, c in model_counts.items():
        print(f"  {m}: {c}")
    
    # Статистика T1/2
    t12_vals = [r["T1/2_best"] for r in all_results if r["T1/2_best"] is not None]
    if t12_vals:
        print(f"\nT1/2 статистика:")
        print(f"  min:  {min(t12_vals):.2f}")
        print(f"  max:  {max(t12_vals):.2f}")
        print(f"  mean: {np.mean(t12_vals):.2f}")
        print(f"  median: {np.median(t12_vals):.2f}")
    
    return all_results


if __name__ == "__main__":
    main()