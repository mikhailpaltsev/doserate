# AGENTS.md — Домашняя работа 2

## Проверка верификатора

```bash
python -m pytest checks -q
```

## Запуск эксперимента

```bash
python run_experiment.py dry      # проверка схемы
python run_experiment.py smoke    # один seed
python run_experiment.py sweep    # все 5 seed
python run_experiment.py full     # полный расчёт с вердиктом
```

Результаты сохраняются в `results/`.