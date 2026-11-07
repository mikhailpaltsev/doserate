"""Одноэкспоненциальная модель мощности дозы после введения I-131.

Модель мощности дозы:
    D(t) = B + A * exp(-lambda * t)

Параметры:
    B = 0.2 мкЗв/ч  -- фиксированный фон (не оценивается)
    A >= 0            -- амплитуда экспоненциального компонента, мкЗв/ч
    lambda > 0        -- коэффициент выведения, ч^-1

Эффективный период полувыведения:
    T_1/2 = ln(2) / lambda   [ч]

Использование
-------------
>>> from src.exponential import dose_rate, fit_exponential, half_life
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit


# ── Константы ──────────────────────────────────────────────────────────

BACKGROUND: float = 0.2  # мкЗв/ч -- фиксированный фон


# ── Типы возвращаемых значений ─────────────────────────────────────────

class FitResult(NamedTuple):
    """Результат подгонки одноэкспоненциальной модели.

    Attributes
    ----------
    A : float
        Оценённая амплитуда, мкЗв/ч.
    lam : float
        Оценённый коэффициент выведения, ч^-1.
    half_life : float
        Эффективный период полувыведения, ч.
    A_std : float | None
        Стандартная ошибка оценки A (None при неудачной подгонке).
    lam_std : float | None
        Стандартная ошибка оценки lambda (None при неудачной подгонке).
    """

    A: float
    lam: float
    half_life: float
    A_std: float | None
    lam_std: float | None


# ── Утилиты ────────────────────────────────────────────────────────────

def _to_float_array(x) -> np.ndarray:
    """Преобразует вход в np.ndarray[float64], проверяя конечность."""
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Входные данные должны содержать только конечные числа")
    return arr


# ── Модель мощности дозы ──────────────────────────────────────────────

def dose_rate(
    t: float | np.ndarray,
    A: float,
    lam: float,
) -> float | np.ndarray:
    """Вычисляет мощность дозы по модели D(t) = B + A*exp(-lambda*t).

    Parameters
    ----------
    t : float or array-like
        Время в часах (t >= 0).
    A : float
        Амплитуда экспоненциального компонента (A >= 0).
    lam : float
        Коэффициент выведения (lambda > 0), ч^-1.

    Returns
    -------
    float or np.ndarray
        Мощность дозы в мкЗв/ч.

    Raises
    ------
    ValueError
        Если A < 0, lam <= 0, или входные значения не являются конечными числами.
    """
    # Проверка конечности и типов
    t_arr = _to_float_array(t)
    if not np.isfinite(A):
        raise ValueError(f"A должно быть конечным числом, получено {A}")
    if not np.isfinite(lam):
        raise ValueError(f"lam должно быть конечным числом, получено {lam}")
    if A < 0:
        raise ValueError(f"A должно быть >= 0, получено {A}")
    if lam <= 0:
        raise ValueError(f"lam должно быть > 0, получено {lam}")
    if np.any(t_arr < 0):
        raise ValueError("Время должно быть >= 0")

    return BACKGROUND + A * np.exp(+lam * t_arr)


def half_life(lam: float) -> float:
    """Эффективный период полувыведения: ln(2) / lambda.

    Parameters
    ----------
    lam : float
        Коэффициент выведения (lambda > 0), ч^-1.

    Returns
    -------
    float
        Период полувыведения в часах.

    Raises
    ------
    ValueError
        Если lambda <= 0 или не является конечным числом.
    """
    if not np.isfinite(lam):
        raise ValueError(f"lambda должно быть конечным числом, получено {lam}")
    if lam <= 0:
        raise ValueError(f"lambda должно быть > 0, получено {lam}")
    return math.log(2) / lam


# ── Подгонка методом нелинейных наименьших квадратов ───────────────────

def _model(t: np.ndarray, A: float, lam: float) -> np.ndarray:
    """Внутренняя функция модели для scipy.optimize.curve_fit.

    Фон B фиксирован и не является параметром подгонки.
    """
    return BACKGROUND + A * np.exp(-lam * t)


def fit_exponential(
    times,
    measurements,
) -> FitResult:
    """Оценка параметров A и lambda методом нелинейных наименьших квадратов.

    Подгоняет модель D(t) = B + A*exp(-lambda*t) к наблюдаемым значениям
    мощности дозы. Фон B фиксирован (= 0.2 мкЗв/ч). Ограничения:
    A >= 0, lambda > 0.

    Parameters
    ----------
    times : array-like
        Массив временных точек в часах (обучающие точки).
        Должно быть не менее 3 уникальных временных точек.
    measurements : array-like
        Массив средних значений мощности дозы (мкЗв/ч) в соответствующих
        временных точках.

    Returns
    -------
    FitResult
        Именованный кортеж с оценками параметров и стандартными ошибками.

    Raises
    ------
    ValueError
        Если входные данные некорректны (пустые, разная длина, нечисловые,
        меньше 3 точек, неконечные значения).
    """
    # Преобразование и проверка конечности
    try:
        times = _to_float_array(times)
        measurements = _to_float_array(measurements)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Входные данные должны быть числовыми: {e}") from e

    # ── Валидация ────────────────────────────────────────────────────
    if times.ndim != 1 or measurements.ndim != 1:
        raise ValueError("times и measurements должны быть одномерными массивами")
    if times.size == 0:
        raise ValueError("times не должен быть пустым")
    if times.shape[0] != measurements.shape[0]:
        raise ValueError(
            f"times ({times.shape[0]}) и measurements "
            f"({measurements.shape[0]}) имеют разную длину"
        )
    if times.shape[0] < 3:
        raise ValueError(
            f"Требуется минимум 3 временные точки, получено {times.shape[0]}"
        )
    if np.any(times < 0):
        raise ValueError("Временные точки должны быть неотрицательными")
    if np.any(measurements < 0):
        raise ValueError("Значения мощности дозы должны быть неотрицательными")

    # ── Начальные оценки ─────────────────────────────────────────────
    # A ~ среднее измерений - фон; lambda ~ 0.01 ч^-1
    p0_A = float(np.mean(measurements) - BACKGROUND)
    p0_A = max(p0_A, 1.0)  # минимальная начальная оценка для устойчивости
    p0_lam = 0.01

    # ── Подгонка ─────────────────────────────────────────────────────
    popt, pcov = curve_fit(
        _model,
        times,
        measurements,
        p0=(p0_A, p0_lam),
        bounds=([0.0, 1e-6], [np.inf, np.inf]),
        maxfev=10000,
    )

    A_est, lam_est = popt

    # ── Стандартные ошибки ───────────────────────────────────────────
    if pcov is not None and np.all(np.diag(pcov) >= 0):
        A_std = float(np.sqrt(pcov[0, 0]))
        lam_std = float(np.sqrt(pcov[1, 1]))
    else:
        A_std = None
        lam_std = None

    return FitResult(
        A=float(A_est),
        lam=float(lam_est),
        half_life=half_life(lam_est),
        A_std=A_std,
        lam_std=lam_std,
    )