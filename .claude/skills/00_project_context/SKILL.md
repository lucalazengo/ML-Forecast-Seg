---
name: ml-forecast-project-context
description: >
  Contexto mestre do projeto ML-Forecast-Seg. SEMPRE leia esta skill primeiro,
  antes de qualquer outra tarefa no projeto. Ela define stack, convenções,
  regras invioláveis e aponta para as skills específicas de cada etapa.
  Use quando: iniciar qualquer tarefa no projeto, precisar de contexto sobre
  o pipeline, entender convenções de nomenclatura, identificar qual skill
  acionar em seguida.
---

# ML-Forecast-Seg — Contexto Mestre do Projeto

## ⚠️ Leia sempre primeiro

Esta skill é o ponto de entrada. Após lê-la, identifique a tarefa e acione
a skill específica correspondente listada em "Mapa de Skills".

---

## Objetivo do Projeto

Previsão de **casos novos** de doenças/eventos usando séries temporais.
Modelos principais: **XGBoost** e **LightGBM** com feature engineering temporal.

| Parâmetro         | Valor padrão       | Ajustar em: |
|-------------------|--------------------|-------------|
| Horizonte         | 7 dias             | `src/config.py` |
| Granularidade     | Diária             | `src/config.py` |
| Variável alvo     | `casos_novos`      | `src/config.py` |
| Coluna de data    | `data`             | `src/config.py` |
| Random state      | 42 (sempre)        | todo lugar  |

---

## Stack

```
Python >= 3.9
pandas, numpy
scikit-learn
xgboost >= 1.7
lightgbm >= 3.3
optuna              # tuning
statsmodels         # testes estatísticos
matplotlib, seaborn # visualização
joblib              # serialização de modelos
```

---

## Estrutura de Diretórios

```
ML-Forecast-Seg/
├── .claude/
│   └── skills/              # Skills do Claude Code
├── src/
│   ├── config.py            # Parâmetros globais
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_xgboost.py
│   │   ├── train_lightgbm.py
│   │   └── evaluate.py
│   └── utils/
│       ├── metrics.py
│       └── validation.py
├── models/                  # Artefatos treinados (.json, .pkl)
├── reports/                 # Outputs, gráficos, métricas
├── dashboard/               # Next.js frontend
├── data/
│   ├── raw/
│   └── processed/
└── notebooks/               # Exploração apenas
```

---

## Regras Invioláveis

1. **Jamais usar dados futuros no treino** — todo shift/lag deve respeitar o horizonte
2. **Validação sempre com `TimeSeriesSplit`** — nunca `KFold` aleatório
3. **`random_state=42`** em todo lugar que aceitar esse parâmetro
4. **Baseline obrigatório** — sempre comparar contra naive antes de reportar resultado
5. **Reversão de transformações** — se aplicou log/diff, reverter antes de calcular métricas finais
6. **Nomear modelos com timestamp** — `modelo_YYYYMMDD_METRICA_VALOR.json`

---

## config.py padrão

```python
# src/config.py
import os

# Caminhos
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

# Série temporal
TARGET_COL = "casos_novos"
DATE_COL = "data"
FREQ = "D"           # "D" diária, "W" semanal
HORIZON = 7          # dias à frente para prever
RANDOM_STATE = 42

# Validação
N_SPLITS = 5         # splits no TimeSeriesSplit
GAP = HORIZON        # gap entre treino e validação

# Features
LAG_LIST = [7, 14, 21, 28]     # definido em função do HORIZON
ROLLING_WINDOWS = [7, 14, 28]
```

---

## Mapa de Skills — Quando usar cada uma

| Tarefa                              | Acionar skill                        |
|-------------------------------------|--------------------------------------|
| Carregar e limpar dados             | `01_data_preprocessing`              |
| Criar features temporais/lags       | `02_feature_engineering`             |
| Detectar/tratar outliers            | `03_outlier_anomaly`                 |
| Treinar/tunar XGBoost               | `04_xgboost_forecast`                |
| Treinar/tunar LightGBM              | `05_lightgbm_forecast`               |
| Avaliar modelo, calcular métricas   | `06_evaluation_metrics`              |
| Organizar scripts, estruturar pipe  | `07_pipeline_structure`              |

---

## Fluxo completo do pipeline

```
raw data
   │
   ▼
[01] preprocess.py      → data/processed/series_clean.parquet
   │
   ▼
[02+03] build_features.py → data/processed/features.parquet
   │
   ├──▶ [04] train_xgboost.py  → models/xgb_YYYYMMDD_rmse_XX.json
   └──▶ [05] train_lightgbm.py → models/lgbm_YYYYMMDD_rmse_XX.pkl
              │
              ▼
           [06] evaluate.py → reports/evaluation_YYYYMMDD.csv
```
