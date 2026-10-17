"""
Улучшенная версия: подбор 3 лучших моделей экспоненциального распада и расчёт T1/2.
Исправления:
  - bounds для curve_fit (λ > 0, A > 0)
  - улучшенная инициализация через log-линейную регрессию
  - надёжный расчёт T1/2 с fallback
"""

import json
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from pathlib import Path

RESULT_DIR = Path(__file__).parent.parent / "result"
PARSED_FILE = RESULT_DIR / "parsed_data.json"

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)


# --- Модели ---

def model_single(t, A0, lam):
    return A0 * np.exp(-lam * t)


def model_double(t, A1, lam1, A2, lam2):
    return A1 * np.exp(-lam1 * t) + A2 * np.exp(-lam2 * t)


def model_triple(t, A1, lam1, A2, lam2, A3, lam3):
    return (A1 * np.exp(-lam1 * t) 
            + A2 * np.exp(-lam2 * t) 
            + A3 * np.exp(-lam3 * t))


MODELS = [
    ("M1_single", model_single, 2),
    ("M2_double", model_double, 4),
    ("M3_triple", model_triple, 6),
]


# --- Метрики ---

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def aic(n, n_params, residuals):
    rss = np.sum(residuals ** 2)
    if rss <= 0 or n <= 0:
        return -1e10  # идеальная подгонка
    return n * np.log(rss / n) + 2 * n_params


# --- Инициализация параметров ---

def init_params_single(t, means):
    """Инициализация через log-линейную регрессию."""
    # log(A) = log(A0) - lam*t => линейная регрессия
    log_means = np.log(means + 1e-10)
    coeffs = np.polyfit(t, log_means, 1)
    lam = -coeffs[0]
    A0 = np.exp(coeffs[1])
    if lam <= 0:
        lam = 0.1
    if A0 <= 0:
        A0 = means[0]
    return [max(A0, 1e-6), max(lam, 1e-6)]


def init_params_double(t, means):
    """Инициализация: быстрая и медленная компонента."""
    # Используем log-линейную регрессию для начальной оценки
    log_means = np.log(means + 1e-10)
    coeffs = np.polyfit(t, log_means, 1)
    lam_fast = max(-coeffs[0], 0.01)
    A0 = max(np.exp(coeffs[1]), means[0])
    
    # Быстрая компонента ~70%, медленная ~30%
    return [A0 * 0.7, lam_fast * 2, A0 * 0.3, lam_fast * 0.1]


def init_params_triple(t, means):
    """Инициализация: три компоненты с убывающими λ."""
    log_means = np.log(means + 1e-10)
    coeffs = np.polyfit(t, log_means, 1)
    lam_base = max(-coeffs[0], 0.01)
    A0 = max(np.exp(coeffs[1]), means[0])
    
    return [
        A0 * 0.5, lam_base * 3,   # быстрая
        A0 * 0.3, lam_base * 1,   # средняя
        A0 * 0.2, lam_base * 0.2, # медленная
    ]


INIT_FUNCS = {
    "M1_single": init_params_single,
    "M2_double": init_params_double,
    "M3_triple": init_params_triple,
}


# --- Bounds ---

def get_bounds(n_params):
    """Границы для curve_fit: все A > 0, λ > 0."""
    if n_params == 2:
        return ([0, 1e-6], [np.inf, 5.0])
    elif n_params == 4:
        return ([0, 1e-6, 0, 1e-6], [np.inf, 5.0, np.inf, 5.0])
    else:  # 6
        return ([0, 1e-6, 0, 1e-6, 0, 1e-6], [np.inf, 5.0, np.inf, 5.0, np.inf, 5.0])


# --- Расчёт T1/2 ---

def calc_t12_single(t, means):
    """T1/2 = ln(2) / λ для однокомпонентной модели."""
    try:
        p0 = init_params_single(t, means)
        bounds = get_bounds(2)
        popt, _ = curve_fit(model_single, t, means, p0=p0, bounds=bounds, maxfev=10000)
        lam = popt[1]
        if lam > 0:
            return float(np.log(2) / lam)
    except Exception:
        pass
    return None


def calc_t12_multi(t, means, model_func, n_params, model_name):
    """
    Эффективный T1/2 — время когда модель падает до половины от начального значения.
    """
    try:
        p0 = INIT_FUNCS[model_name](t, means)
        bounds = get_bounds(n_params)
        popt, _ = curve_fit(model_func, t, means, p0=p0, bounds=bounds, maxfev=10000)
        
        # Проверяем что все λ > 0
        for i in range(1, n_params, 2):
            if popt[i] <= 0 or popt[i] > 5:
                return None
        
        # Вычисляем модель на плотной сетке
        t_max = max(t[-1], t[-1] * 3)  # экстраполируем до 3x последнего времени
        t_fine = np.linspace(t[0], t_max, 5000)
        y_fine = model_func(t_fine, *popt)
        
        # Проверяем что значения конечны
        if not np.all(np.isfinite(y_fine)):
            return None
        
        A_start = means[0]
        A_target = A_start / 2
        
        # Ищем пересечение
        for i in range(1, len(y_fine)):
            if y_fine[i] <= A_target and y_fine[i-1] > A_target:
                frac = (A_target - y_fine[i-1]) / (y_fine[i] - y_fine[i-1] + 1e-15)
                return float(t_fine[i-1] + frac * (t_fine[i] - t_fine[i-1]))
        
        # Если не достигли половины — экстраполируем
        if y_fine[-1] > 0:
            # Используем последнюю точку для оценки
            last_lam = popt[-1]  # самая медленная компонента
            if last_lam > 0:
                # A(t) = A_last * exp(-last_lam * (t - t_last))
                # A_last / 2 = A_last * exp(-last_lam * dt)
                # dt = ln(2) / last_lam
                return float(t_fine[-1] + np.log(2) / last_lam)
        
        return None
    except Exception:
        return None


# --- Основной цикл ---

def fit_patient(patient: dict) -> dict:
    """Фитит все 3 модели для одного пациента."""
    t = np.array(patient["times"])
    means = np.array(patient["means"])
    n = len(t)
    
    results = {}
    
    for name, model_func, n_params in MODELS:
        try:
            p0 = INIT_FUNCS[name](t, means)
            bounds = get_bounds(n_params)
            
            popt, pcov = curve_fit(
                model_func, t, means, p0=p0, bounds=bounds, maxfev=10000
            )
            
            y_pred = model_func(t, *popt)
            residuals = means - y_pred
            
            r2 = r_squared(means, y_pred)
            rmse_val = rmse(means, y_pred)
            aic_val = aic(n, n_params, residuals)
            
            # T1/2
            if name == "M1_single":
                t12 = calc_t12_single(t, means)
            else:
                t12 = calc_t12_multi(t, means, model_func, n_params, name)
            
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
        
        row = {"patient_id": pid}
        
        for rank, (mname, aic_val, t12_val) in enumerate(top3, 1):
            row[f"T1/2_model_{rank}"] = round(t12_val, 4) if t12_val is not None else None
            row[f"R2_model_{rank}"] = fit[mname]["R2"]
            row[f"RMSE_model_{rank}"] = fit[mname]["RMSE"]
            row[f"AIC_model_{rank}"] = round(aic_val, 4)
            row[f"best_model_{rank}"] = mname
        
        row["T1/2_best"] = round(best_t12, 4) if best_t12 is not None else None
        all_results.append(row)
    
    print(f"Обработано: {len(all_results)}, пропущено: {skipped}")
    
    # Сохраняем JSON
    out_json = RESULT_DIR / "results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"JSON сохранён: {out_json}")
    
    # Сохраняем Excel
    import pandas as pd
    df = pd.DataFrame(all_results)
    out_xlsx = RESULT_DIR / "results.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"Excel сохранён: {out_xlsx}")
    
    # Статистика
    model_counts = {"M1_single": 0, "M2_double": 0, "M3_triple": 0}
    for r in all_results:
        for k in ["best_model_1", "best_model_2", "best_model_3"]:
            m = r.get(k)
            if m:
                model_counts[m] = model_counts.get(m, 0) + 1
    
    print(f"\nРаспределение лучших моделей:")
    for m, c in model_counts.items():
        print(f"  {m}: {c}")
    
    t12_vals = [r["T1/2_best"] for r in all_results if r["T1/2_best"] is not None]
    if t12_vals:
        print(f"\nT1/2 статистика (n={len(t12_vals)}):")
        print(f"  min:  {min(t12_vals):.2f} ч")
        print(f"  max:  {max(t12_vals):.2f} ч")
        print(f"  mean: {np.mean(t12_vals):.2f} ч")
        print(f"  median: {np.median(t12_vals):.2f} ч")
    
    # Проверка NaN
    nan_count = sum(1 for r in all_results if r["T1/2_best"] is None)
    print(f"  NaN: {nan_count}/{len(all_results)}")
    
    return all_results


if __name__ == "__main__":
    main()