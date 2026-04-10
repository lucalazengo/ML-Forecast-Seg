---
name: ts-lightgbm-forecast
description: >
  Treinamento, validação e tuning especialista de LightGBM para previsão de
  séries temporais no projeto ML-Forecast-Seg. Use esta skill quando: configurar
  num_leaves e min_data_in_leaf (os parâmetros mais críticos do LGBM), usar
  callbacks de early stopping corretamente, tunar com Optuna + pruning, comparar
  LGBM vs XGBoost, ou entender diferenças de crescimento leaf-wise vs level-wise.
  Sempre usar após skill de feature engineering.
---

# Skill: LightGBM para Previsão de Séries Temporais

## Script pronto

```bash
python .claude/skills/05_lightgbm_forecast/scripts/train_lightgbm.py
```

---

## LightGBM vs XGBoost — quando usar cada um

| Critério                  | LightGBM                    | XGBoost                      |
|---------------------------|-----------------------------|------------------------------|
| **Dataset grande** (>50k) | ✅ Muito mais rápido         | Mais lento                   |
| **Dataset pequeno** (<5k) | Cuidado com overfit         | ✅ Mais robusto               |
| **Muitas features**       | ✅ Feature fraction eficiente | OK                          |
| **Interpretabilidade**    | Similar                     | Similar                      |
| **Tuning**                | `num_leaves` é o principal  | `max_depth` é o principal    |
| **Memória**               | ✅ Menor consumo             | Maior consumo                |

**Para séries epidemiológicas com granularidade diária e poucos anos:**
Testar ambos via cross-validation e escolher pelo RMSE médio.

---

## Diferença fundamental: leaf-wise vs level-wise

```
XGBoost (level-wise):           LightGBM (leaf-wise):
Cresce camada por camada        Cresce na folha com maior ganho
[menos overfit naturalmente]    [mais rápido, mas overfit com num_leaves alto]

         Raiz                            Raiz
        /    \                          /    \
      N1      N2          →           N1      N2
     /  \    /  \                    /  \
   N3   N4  N5  N6                 N3   N4 ← expande aqui (maior ganho)
```

**Consequência prática:** `num_leaves` no LGBM precisa ser controlado.
Uma árvore com `max_depth=5` tem no máximo 32 folhas, mas LGBM pode criar
muito mais se `num_leaves` for alto.

---

## Parâmetros base — ponto de partida seguro

```python
params = {
    'n_estimators'     : 1000,
    'learning_rate'    : 0.05,
    'num_leaves'       : 31,       # ← MAIS IMPORTANTE do LGBM
    'max_depth'        : -1,       # deixar -1 (controlado por num_leaves)
    'min_data_in_leaf' : 20,       # ← SEGUNDO MAIS IMPORTANTE
    'feature_fraction' : 0.8,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 5,
    'lambda_l1'        : 0.05,
    'lambda_l2'        : 1.0,
    'min_gain_to_split': 0.01,
    'random_state'     : 42,
    'verbose'          : -1,       # silencia output
}
```

### Guia de ajuste fino

| Parâmetro           | Aumentar quando...             | Diminuir quando...            |
|---------------------|--------------------------------|-------------------------------|
| `num_leaves`        | Underfitting                   | Overfitting (mais comum)      |
| `min_data_in_leaf`  | Série ruidosa / dados escassos | Dataset grande, underfitting  |
| `feature_fraction`  | Underfitting                   | Overfitting, muitas features  |
| `lambda_l1`         | Muitas features irrelevantes   | Poucos dados                  |
| `learning_rate`     | Treinamento muito lento        | Overfitting / instabilidade   |

**Regra de ouro LGBM:** `num_leaves < 2^max_depth`
Para max_depth equivalente = 5 → `num_leaves ≤ 31` (conservador)

---

## Callbacks — sintaxe correta (LGBM >= 3.3)

```python
import lightgbm as lgb

# CORRETO — usar callbacks, não parâmetros diretos
callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=-1),   # -1 = silenciar, 50 = logar a cada 50 iter
]

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=callbacks
)

# ❌ ERRADO — API antiga descontinuada:
# model.fit(..., early_stopping_rounds=50, verbose=50)
```

---

## Validação temporal — OBRIGATÓRIO

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=HORIZON)

for train_idx, val_idx in tscv.split(X):
    X_tr,  y_tr  = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx],   y.iloc[val_idx]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    best_iter = model.best_iteration_   # usar no treino final
```

---

## Tuning com Optuna + Pruning

```python
import optuna

# Pruner elimina trials ruins cedo — economiza tempo
pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=pruner
)
study.optimize(objective, n_trials=50)
```

**Espaço de busca recomendado para LGBM em séries temporais:**

```python
params = {
    'num_leaves'       : trial.suggest_int('num_leaves', 15, 200),
    'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 5, 100),
    'learning_rate'    : trial.suggest_float('lr', 0.005, 0.3, log=True),
    'feature_fraction' : trial.suggest_float('ff', 0.5, 1.0),
    'bagging_fraction' : trial.suggest_float('bf', 0.5, 1.0),
    'lambda_l1'        : trial.suggest_float('l1', 1e-4, 10, log=True),
    'lambda_l2'        : trial.suggest_float('l2', 1e-4, 10, log=True),
}
```

---

## Feature importance — gain vs split

```python
# 'gain' — quanto cada feature contribuiu para reduzir o erro
# PREFERÍVEL: mais significativo que simples contagem de splits
importance_gain = pd.Series(
    model.feature_importances_,   # padrão do scikit-learn API = split
    index=feature_cols
)

# Para gain explicitamente via booster:
booster = model.booster_
importance_gain = pd.Series(
    booster.feature_importance(importance_type='gain'),
    index=feature_cols
).sort_values(ascending=False)
```

---

## Treino final sem early stopping

```python
# Usar o best_iteration_ médio dos folds de CV
avg_best_iter = int(np.mean([m['best_iter'] for m in fold_results]))

params_final = {**best_params}
params_final['n_estimators'] = avg_best_iter
# NÃO incluir early_stopping nos callbacks do treino final

model_final = lgb.LGBMRegressor(**params_final)
model_final.fit(X_full, y_full, callbacks=[lgb.log_evaluation(-1)])
```

---

## Erros comuns

- ❌ `num_leaves` muito alto (> 100) sem `min_data_in_leaf` alto — overfit severo
- ❌ Usar API antiga de early stopping (`early_stopping_rounds` como param) — descontinuada
- ❌ `verbose=-1` não colocado — inunda o terminal com logs
- ❌ Importance type = 'split' — infla features com muitos valores únicos
- ✅ Sempre testar LGBM vs XGBoost via CV antes de escolher modelo final
- ✅ `bagging_freq > 0` obrigatório para `bagging_fraction` funcionar
