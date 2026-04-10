---
name: ts-pipeline-structure
description: >
  Estruturação e organização do pipeline de scripts Python no projeto
  ML-Forecast-Seg. Use esta skill quando: reorganizar scripts soltos em
  módulos, criar config.py central, definir contratos entre etapas do pipeline,
  implementar logging padronizado, estruturar para reprodutibilidade, ou criar
  o script run_pipeline.py que orquestra todas as etapas.
---

# Skill: Estrutura de Pipeline ML-Forecast-Seg

## Estrutura alvo de diretórios

```
ML-Forecast-Seg/
├── src/
│   ├── config.py                 ← parâmetros globais centralizados
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py         ← skill 01
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py     ← skill 02
│   │   └── outlier_detection.py  ← skill 03
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_xgboost.py      ← skill 04
│   │   ├── train_lightgbm.py     ← skill 05
│   │   └── evaluate.py           ← skill 06
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── validation.py
├── run_pipeline.py               ← orquestrador principal
├── data/
│   ├── raw/                      ← dados brutos (nunca modificar)
│   └── processed/                ← parquets intermediários
├── models/                       ← artefatos treinados
├── reports/                      ← métricas, gráficos, csvs
├── notebooks/                    ← exploração apenas, nunca produção
└── .claude/skills/               ← skills do Claude Code
```

---

## run_pipeline.py — orquestrador completo

```python
"""
run_pipeline.py
Executa o pipeline completo de forecast de ponta a ponta.
Uso: python run_pipeline.py [--skip-tuning] [--model xgb|lgbm|both]
"""
import argparse
import logging
import time
from src.config import *
from src.data.preprocess import preprocess_pipeline
from src.features.build_features import build_all_features, get_feature_columns
from src.features.outlier_detection import full_outlier_pipeline
from src.models.train_xgboost import cross_validate_xgboost, train_final_xgboost, save_model as save_xgb
from src.models.train_lightgbm import cross_validate_lightgbm, train_final_lightgbm, save_model as save_lgbm
from src.models.evaluate import evaluate_baseline, generate_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(f'reports/pipeline_{time.strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pipeline')


def run(args):
    start = time.time()
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE ML-FORECAST-SEG")
    logger.info("=" * 60)

    # ── ETAPA 1: Pré-processamento ────────────────────────────
    logger.info("[1/5] Pré-processamento...")
    df, inverse_fn = preprocess_pipeline(
        filepath="data/raw/casos.csv",
        transformation='none'  # trocar por 'log' se série não estacionária
    )
    df.to_parquet("data/processed/series_clean.parquet")

    # ── ETAPA 2: Outliers ─────────────────────────────────────
    logger.info("[2/5] Detecção e tratamento de outliers...")
    df = full_outlier_pipeline(df)
    # Usar série limpa para features
    df[TARGET_COL] = df[f'{TARGET_COL}_clean']
    df.to_parquet("data/processed/series_outlier_treated.parquet")

    # ── ETAPA 3: Feature engineering ──────────────────────────
    logger.info("[3/5] Feature engineering...")
    df = build_all_features(df)
    df.to_parquet("data/processed/features.parquet")
    feature_cols = get_feature_columns(df)
    logger.info(f"Features criadas: {len(feature_cols)}")

    # ── ETAPA 4: Baseline ─────────────────────────────────────
    logger.info("[4/5] Calculando baseline...")
    baseline = evaluate_baseline(df)

    # ── ETAPA 5: Modelagem ────────────────────────────────────
    logger.info("[5/5] Treinando modelos...")
    results = {}

    if args.model in ('xgb', 'both'):
        logger.info("  → XGBoost...")
        cv_xgb = cross_validate_xgboost(df, feature_cols)
        model_xgb = train_final_xgboost(df, feature_cols)
        save_xgb(model_xgb, cv_xgb, feature_cols)
        results['XGBoost'] = cv_xgb

    if args.model in ('lgbm', 'both'):
        logger.info("  → LightGBM...")
        cv_lgbm = cross_validate_lightgbm(df, feature_cols)
        model_lgbm = train_final_lightgbm(df, feature_cols)
        save_lgbm(model_lgbm, cv_lgbm, feature_cols)
        results['LightGBM'] = cv_lgbm

    # ── Relatório final ───────────────────────────────────────
    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info("RESULTADOS FINAIS")
    logger.info(f"{'='*60}")
    for name, cv in results.items():
        logger.info(f"{name}: RMSE={cv['rmse_mean']:.2f} ± {cv['rmse_std']:.2f} | MAPE={cv['mape_mean']:.2%}")
    logger.info(f"Baseline Naive RMSE: {baseline['naive']['rmse']:.2f}")
    logger.info(f"Tempo total: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='both', choices=['xgb', 'lgbm', 'both'])
    parser.add_argument('--skip-tuning', action='store_true')
    args = parser.parse_args()
    run(args)
```

---

## config.py — parâmetros centralizados

```python
# src/config.py — ÚNICA fonte da verdade para parâmetros
import os

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

TARGET_COL = "casos_novos"
DATE_COL   = "data"
FREQ       = "D"
HORIZON    = 7
RANDOM_STATE = 42
N_SPLITS   = 5
GAP        = HORIZON

# Lags: nunca menor que HORIZON
LAG_LIST        = [HORIZON, HORIZON+7, HORIZON+14, HORIZON+21]
ROLLING_WINDOWS = [7, 14, 28]
```

---

## Convenção de nomenclatura de artefatos

```
models/
├── xgb_20250410_1430_rmse_12.34.json        # modelo XGBoost
├── xgb_20250410_1430_rmse_12.34_meta.json   # metadados
├── lgbm_20250410_1445_rmse_11.89.pkl        # modelo LGBM
└── lgbm_20250410_1445_rmse_11.89_meta.json

reports/
├── pipeline_20250410_1430.log               # log da execução
├── evaluation_20250410.csv                  # métricas comparativas
├── diagnostico_serie.png                    # análise exploratória
├── outliers.png                             # outliers detectados
└── evaluation_panel.png                     # painel de avaliação
```

---

## Checklist de reprodutibilidade

- [ ] `config.py` centralizado com todos os parâmetros
- [ ] `random_state=42` em todo lugar
- [ ] Parquets intermediários salvos em `data/processed/`
- [ ] Metadados JSON junto com cada modelo salvo
- [ ] Log de cada execução em `reports/`
- [ ] `run_pipeline.py` executa tudo do zero sem intervenção manual
- [ ] `requirements.txt` com versões fixadas (`pip freeze > requirements.txt`)

---

## Erros comuns de organização

- ❌ Parâmetros hardcoded espalhados pelos scripts
- ❌ Scripts que dependem de variáveis globais de notebooks
- ❌ Modelos salvos sem metadados (impossível reproduzir)
- ❌ `data/raw/` sendo modificado (dados brutos são imutáveis)
- ✅ Cada script pode ser executado de forma independente
- ✅ Todo script loga início, fim e métricas principais
