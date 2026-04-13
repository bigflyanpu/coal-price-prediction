# Core Unified Project

This folder is the canonical runtime root.

## Run training

```bash
cd core
python train.py
```

## Run service

```bash
cd core
gunicorn app:app --bind 0.0.0.0:7860 --timeout 120
```

## Runtime directories

- `models/`: trained artifacts (generated at runtime)
- `reports/`: evaluation reports (generated at runtime)
- `data/`: runtime data layers (generated/managed at runtime)
