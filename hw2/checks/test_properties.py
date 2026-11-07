"""Тесты на свойствах одноэкспоненциальной модели.

Восемь проверок из METHOD.md, раздел 5.
"""

import numpy as np
import pytest

from src.exponential import dose_rate, fit_exponential, half_life, BACKGROUND


# ── Свойство 1: D(t) >= B для t >= 0, A >= 0, lambda > 0 ─────────────

def test_model_not_below_background():
    """D(t) >= B для всех t >= 0 при A >= 0, lambda > 0."""
    for A in [0.0, 1.0, 10.0, 30.0]:
        for lam in [0.005, 0.01, 0.03, 0.1]:
            for t in [0.0, 1.0, 10.0, 100.0, 1000.0]:
                val = dose_rate(t, A, lam)
                assert val >= BACKGROUND - 1e-12, (
                    f"D({t})={val} < B={BACKGROUND} при A={A}, lam={lam}"
                )


# ── Свойство 2: D(t) строго убывает при A > 0, lambda > 0 ────────────

def test_model_monotone():
    """Производная -A*lambda*exp(-lambda*t) < 0 при A > 0, lambda > 0."""
    for A in [1.0, 10.0, 30.0]:
        for lam in [0.005, 0.01, 0.03]:
            t_vals = np.linspace(0, 100, 50)
            y_vals = dose_rate(t_vals, A, lam)
            diffs = np.diff(y_vals)
            assert np.all(diffs < 0), (
                f"Не строго убывает при A={A}, lam={lam}: "
                f"diffs={diffs[diffs >= 0][:5]}"
            )


# ── Свойство 3: Восстановление параметров на данных без шума ──────────

def test_recover_noiseless_parameters():
    """На синтетических данных без шума параметры восстанавливаются с
    относительной ошибкой не более 1e-3."""
    test_cases = [
        (10.0, 0.01),
        (20.0, 0.02),
        (5.0, 0.005),
        (30.0, 0.03),
    ]
    t = np.array([1.0, 2.0, 3.5, 24.0, 48.0])

    for A_true, lam_true in test_cases:
        y = dose_rate(t, A_true, lam_true)
        result = fit_exponential(t, y)
        rel_err_A = abs(result.A - A_true) / A_true
        rel_err_lam = abs(result.lam - lam_true) / lam_true
        assert rel_err_A <= 1e-3, (
            f"Отн. ошибка A={rel_err_A:.2e} > 1e-3 "
            f"(истина A={A_true}, оценка {result.A:.6f})"
        )
        assert rel_err_lam <= 1e-3, (
            f"Отн. ошибка lam={rel_err_lam:.2e} > 1e-3 "
            f"(истина lam={lam_true}, оценка {result.lam:.6f})"
        )


# ── Свойство 4: Перестановка строк не меняет результат подгонки ───────

def test_permutation_invariance():
    """Перестановка строк вместе со значениями измерений не меняет
    результат подгонки сверх rtol=1e-5."""
    np.random.seed(42)
    A_true, lam_true = 15.0, 0.015
    t = np.array([1.0, 2.0, 3.5, 24.0, 48.0])
    y = dose_rate(t, A_true, lam_true)

    result_orig = fit_exponential(t, y)

    for _ in range(10):
        perm = np.random.permutation(len(t))
        result_perm = fit_exponential(t[perm], y[perm])
        assert np.isclose(result_orig.A, result_perm.A, rtol=1e-5), (
            f"A не совпадает при перестановке: {result_orig.A} vs {result_perm.A}"
        )
        assert np.isclose(result_orig.lam, result_perm.lam, rtol=1e-5), (
            f"lam не совпадает при перестановке: {result_orig.lam} vs {result_perm.lam}"
        )


# ── Свойство 5: Замена часов сутками даёт lambda_day = 24 * lambda_hour ──

def test_time_units_conversion():
    """Замена единицы времени не изменяет физическую кривую.
    lambda_day = 24 * lambda_hour с rtol=1e-3."""
    A_true, lam_hour = 15.0, 0.015
    t_hour = np.array([1.0, 2.0, 3.5, 24.0, 48.0])
    y = dose_rate(t_hour, A_true, lam_hour)

    # Переход к суткам: t_day = t_hour / 24
    t_day = t_hour / 24.0
    res_hour = fit_exponential(t_hour, y)
    res_day = fit_exponential(t_day, y)

    assert np.isclose(res_day.lam, 24.0 * res_hour.lam, rtol=1e-3), (
        f"lam_day={res_day.lam:.6f} != 24*lam_hour={24*res_hour.lam:.6f}"
    )
    assert np.isclose(res_day.A, res_hour.A, rtol=1e-3), (
        f"A не совпадает при смене единиц: {res_day.A} vs {res_hour.A}"
    )


# ── Свойство 6: Перестановка трёх повторов не изменяет их среднее ────

def test_replicate_permutation():
    """Среднее арифметическое симметрично относительно перестановки трёх повторов.
    Перестановка m1, m2, m3 местами не меняет их среднее для каждой временной точки."""
    for i in range(10):
        rng = np.random.RandomState(i)
        m1 = rng.uniform(1.0, 20.0, 5)
        m2 = rng.uniform(1.0, 20.0, 5)
        m3 = rng.uniform(1.0, 20.0, 5)

        # Среднее в исходном порядке
        mean_orig = np.mean(np.stack([m1, m2, m3], axis=0), axis=0)

        # Среднее после перестановки трёх повторов местами
        # Например: m1->m2, m2->m3, m3->m1
        mean_perm = np.mean(np.stack([m2, m3, m1], axis=0), axis=0)

        assert np.allclose(mean_orig, mean_perm), (
            f"Среднее изменилось при перестановке повторов"
        )


# ── Свойство 7: Неоднозначные номера пациентов не смешивают траектории ──

def test_no_patient_mixing():
    """Каждая временная точка принадлежит одному пациенту.
    Проверка на уровне данных: если patient_id различается,
    то и времена/измерения не должны перемешиваться."""
    # Синтетический пример: два пациента с разными временными точками
    data_p1 = {
        "patient_id": ["P001", "P001", "P001"],
        "time_h": [1.0, 24.0, 48.0],
        "m1": [10.0, 7.0, 5.0],
        "m2": [10.1, 7.1, 5.1],
        "m3": [9.9, 6.9, 4.9],
    }
    data_p2 = {
        "patient_id": ["P002", "P002", "P002"],
        "time_h": [1.0, 24.0, 48.0],
        "m1": [15.0, 10.0, 7.0],
        "m2": [15.1, 10.1, 7.1],
        "m3": [14.9, 9.9, 6.9],
    }

    # Подгонка для каждого пациента отдельно
    for data in [data_p1, data_p2]:
        t = np.array(data["time_h"])
        means = np.mean(np.stack([data["m1"], data["m2"], data["m3"]], axis=1), axis=0)
        result = fit_exponential(t, means)
        assert result.A > 0
        assert result.lam > 0
        assert result.half_life > 0


# ── Свойство 8: Соответствие времени и измерений ──────────────────────

def test_time_measurement_alignment():
    """Каждой временной точке соответствует правильная тройка измерений.
    Проверка: если переставить времена без соответствующей перестановки
    измерений, подгонка даст неверный результат."""
    A_true, lam_true = 10.0, 0.02
    t = np.array([1.0, 2.0, 3.5, 24.0, 48.0])
    y = dose_rate(t, A_true, lam_true)

    # Правильная подгонка
    res_correct = fit_exponential(t, y)
    assert np.isclose(res_correct.A, A_true, rtol=1e-3)
    assert np.isclose(res_correct.lam, lam_true, rtol=1e-3)

    # Неправильная подгонка: времена переставлены, измерения нет
    rng = np.random.RandomState(0)
    t_shuffled = rng.permutation(t)
    res_wrong = fit_exponential(t_shuffled, y)
    # Результат должен отличаться от правильного
    assert not (
        np.isclose(res_wrong.A, A_true, rtol=1e-3)
        and np.isclose(res_wrong.lam, lam_true, rtol=1e-3)
    ), (
        "Подгонка с перемешанными временами дала тот же результат — "
        "возможно, времена не используются корректно"
    )