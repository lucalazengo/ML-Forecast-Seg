---
name: ts-xgboost-forecast
description: >
  Treinamento, validação e tuning especialista de XGBoost para previsão de
  séries temporais no projeto ML-Forecast-Seg. Use esta skill quando: configurar
  parâmetros do XGBoost para forecast, implementar early stopping com validação
  temporal, tunar com Optuna, salvar modelos com metadados, ou analisar feature
  importance. Sempre usar após skill de feature engineering.
---

# Skill: XGBoost para Previsão de Séries Temporais

## Script pronto

```bash
python .claude/skills/04_xgboost_forecast/scripts/train_xgboost.py
```

---

## Parâmetros base — ponto de partida seguro

```python
params = {
    'n_estimators'      : 1000,      # alto + early stopping controla
    'learning_rate'     : 0.05,      # conservador — generaliza melhor
    'max_depth'         : 5,         # séries temporais raramente precisam > 6
    'min_child_weight'  : 3,         # regulariza splits com poucos dados
    'subsample'         : 0.8,       # bagging de linhas
    'colsample_bytree'  : 0.8,       # bagging de features por árvore
    'colsample_bylevel' : 0.8,       # bagging por nível da árvore
    'gamma'             : 0.1,       # custo mínimo para criar split
    'reg_alpha'         : 0.05,      # L1 — induz esparsidade
    'reg_lambda'        : 1.0,       # L2 — suaviza pesos
    'early_stopping_rounds': 50,
    'eval_metric'       : 'rmse',
    'random_state'      : 42,
}
```

### Guia de ajuste fino dos parâmetros mais sensíveis

| Parâmetro          | Valor baixo            | Valor alto               | Problema alvo              |
|--------------------|------------------------|--------------------------|----------------------------|
| `max_depth`        | Underfitting           | Overfitting              | Ajustar primeiro           |
| `learning_rate`    | Mais robusto, mais lento | Rápido, mais frágil     | Sempre < 0.1 para produção |
| `min_child_weight` | Mais splits (overfit)  | Menos splits (underfit)  | Aumentar se série ruidosa  |
| `gamma`            | Mais árvores           | Árvores mais conservadoras | Aumentar se overfitting   |
| `reg_alpha`        | Mais features usadas   | Menos features usadas    | Aumentar se muitas features |

---

## Validação temporal — OBRIGATÓRIO

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5,
    gap=HORIZON   # gap evita que validação "veja" dados do futuro
)

for train_idx, val_idx in tscv.split(X):
    X_tr,  y_tr  = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx],   y.iloc[val_idx]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],   # early stopping monitora este val set
        verbose=False
    )
```

**Por que `gap=HORIZON`?**
Sem o gap, o fold de validação começa imediatamente após o treino.
Com `gap=HORIZON`, simulamos o cenário real onde há um intervalo de
`HORIZON` dias entre o último dado disponível e o dia a prever.

---

## Tuning com Optuna

```python
import optuna

def objective(trial):
    params = {
        'max_depth'        : trial.suggest_int('max_depth', 3, 9),
        'learning_rate'    : trial.suggest_float('lr', 0.005, 0.3, log=True),
        'n_estimators'     : trial.suggest_int('n_est', 200, 2000),
        'subsample'        : trial.suggest_float('sub', 0.5, 1.0),
        'colsample_bytree' : trial.suggest_float('col', 0.5, 1.0),
        'min_child_weight' : trial.suggest_int('mcw', 1, 10),
        'gamma'            : trial.suggest_float('gamma', 0, 1.0),
        'reg_alpha'        : trial.suggest_float('alpha', 1e-4, 10, log=True),
        'reg_lambda'       : trial.suggest_float('lambda', 1e-4, 10, log=True),
        'early_stopping_rounds': 30,
        'random_state'     : 42,
    }
    # Sempre validar com TimeSeriesSplit dentro do objective
    ...
    return mean_rmse_across_folds

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=50)
best_params = study.best_params
```

**Recomendação de n_trials:**
- Exploração inicial: 30 trials
- Refinamento: 50–100 trials
- Produção: 100+ trials com pruning habilitado

---

## Feature importance — análise pós-treino

```python
import pandas as pd

importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

# Regra: manter features com importância > 0.01
selected = importance[importance > 0.01].index.tolist()
print(f"Features selecionadas: {len(selected)}/{len(feature_cols)}")

# Plotar top 20
importance.head(20).plot(kind='barh', figsize=(10, 8))
```

**O que esperar em séries epidemiológicas:**
- `lag_7` e `lag_14` geralmente dominam (sazonalidade semanal)
- `rolling_mean_28` forte em séries com tendência
- `dia_semana_sin/cos` relevante se há padrão semanal de notificação
- Features com importância ≈ 0 devem ser removidas (ruído)

---

## Treino final (sem early stopping)

```python
# Após definir melhores params via CV:
params_final = {**best_params}
params_final.pop('early_stopping_rounds', None)  # remove — não há val set

model_final = xgb.XGBRegressor(**params_final)
model_final.fit(X_full, y_full)   # treina em TODO o dataset
```

---

## Checklist pré-treino

- [ ] Features criadas com lag ≥ horizon (sem leakage)
- [ ] `TimeSeriesSplit` configurado com `gap=HORIZON`
- [ ] Baseline naive calculado para comparação
- [ ] `random_state=42` em todos os params
- [ ] Dataset sem NaN (checar com `df.isnull().sum()`)

---

## Erros comuns

- ❌ `KFold` aleatório — embaralha o tempo, métricas irreais
- ❌ Tuning sem `gap` no TimeSeriesSplit — otimiza para cenário que não existe
- ❌ `early_stopping` no treino final — para antes do tempo sem val set
- ❌ Salvar modelo sem metadados — impossível reproduzir depois
- ✅ Sempre comparar resultado do XGBoost com `model_naive` (último valor)
