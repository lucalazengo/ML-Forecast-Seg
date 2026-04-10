---
name: ts-evaluation-metrics
description: >
  Avaliação especialista de modelos de forecast no projeto ML-Forecast-Seg.
  Use esta skill quando: calcular RMSE, MAE, MAPE, SMAPE, MASE, implementar
  walk-forward validation, calcular baseline naive, analisar resíduos com
  Ljung-Box, detectar viés sistemático, ou gerar relatório comparativo de
  modelos. Usar sempre após treinar qualquer modelo antes de tomar decisões.
---

# Skill: Avaliação de Modelos de Forecast

## Script pronto

```bash
python .claude/skills/06_evaluation_metrics/scripts/evaluate.py
```

---

## Métricas e quando usar cada uma

| Métrica  | Fórmula resumida         | Use quando...                          | Cuidado com...                     |
|----------|--------------------------|----------------------------------------|------------------------------------|
| **RMSE** | √mean((y-ŷ)²)            | Sempre (métrica primária)              | Sensível a outliers                |
| **MAE**  | mean(|y-ŷ|)              | Comunicar resultado para não-técnicos  | Menos sensível a erros grandes     |
| **MAPE** | mean(|y-ŷ|/y)            | Comparar entre séries de escalas diferentes | Explode quando y ≈ 0          |
| **SMAPE**| mean(|y-ŷ|/((y+ŷ)/2))   | Quando há zeros na série              | Assimétrico (sub vs superestimação)|
| **MASE** | MAE / MAE_naive_sazonal  | Benchmark relativo ao naive            | Requer y_train para calcular       |

**Ordem de prioridade no ML-Forecast-Seg:** RMSE → MAE → MAPE

---

## Baseline — sempre calcular antes de reportar

```python
# NUNCA reportar métricas do modelo sem comparar com o naive
# Se o modelo não bate o naive → problema fundamental (provavelmente leakage ou feature engineering)

# Naive: repete último valor observado
naive_pred = y.shift(HORIZON)

# Seasonal naive: repete valor de 7 dias atrás
seasonal_naive_pred = y.shift(7)

# Calcular métricas de ambos
baseline_rmse = rmse(y_valid, naive_pred[valid_idx])
modelo_rmse   = rmse(y_valid, model_pred)

melhora = (baseline_rmse - modelo_rmse) / baseline_rmse * 100
print(f"Melhora vs naive: {melhora:.1f}%")
```

**Se MASE > 1.0:** o modelo é pior que o naive sazonal. Parar e revisar.

---

## Walk-forward validation — simulando o mundo real

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(
    n_splits=5,
    gap=HORIZON    # fundamental: simula intervalo real entre dado e previsão
)

results = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    # Re-treina do zero a cada fold (simula retreino periódico)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    preds = model.predict(X.iloc[test_idx])

    metrics = compute_all_metrics(y.iloc[test_idx], preds)
    metrics['fold'] = fold
    results.append(metrics)

df_results = pd.DataFrame(results)
print(f"RMSE: {df_results['rmse'].mean():.2f} ± {df_results['rmse'].std():.2f}")
```

**Por que re-treinar a cada fold?**
Simula o processo real onde o modelo é retreinado periodicamente.
Usar `.fit()` apenas no primeiro fold e `.predict()` nos demais
superestima a performance.

---

## Análise de resíduos — diagnóstico do modelo

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = y_true - y_pred

# Teste Ljung-Box — detecta autocorrelação nos erros
lb = acorr_ljungbox(residuals, lags=14, return_df=True)
has_autocorr = (lb['lb_pvalue'] < 0.05).any()
```

### O que os resíduos revelam

| Padrão nos resíduos         | Diagnóstico                          | Ação                              |
|-----------------------------|--------------------------------------|-----------------------------------|
| Autocorrelação em lag 7     | Sazonalidade semanal não capturada   | Adicionar lag_7, is_fim_semana    |
| Resíduos crescentes no tempo| Heteroscedasticidade / tendência     | Log-transform da série            |
| Viés positivo sistemático   | Modelo subestima picos               | Feature de surto, weights por erro|
| Resíduos aleatórios         | ✅ Modelo captura toda a estrutura   | Pronto para produção              |

---

## Comparando XGBoost vs LightGBM

```python
results = {}
for name, model in [('XGBoost', xgb_model), ('LightGBM', lgbm_model)]:
    cv = walk_forward_evaluate(df, model, feature_cols)
    results[name] = {
        'rmse_mean': cv['rmse'].mean(),
        'rmse_std' : cv['rmse'].std(),
        'mape_mean': cv['mape'].mean(),
    }

comparison = pd.DataFrame(results).T
print(comparison.sort_values('rmse_mean'))
```

---

## Relatório mínimo de avaliação

Todo resultado deve incluir:

```
Modelo: XGBoost / LightGBM
Horizon: X dias
Período de teste: YYYY-MM-DD a YYYY-MM-DD
N folds CV: 5

RMSE: XX.XX ± XX.XX
MAE:  XX.XX
MAPE: XX.X%
MASE: X.XX (< 1.0 = melhor que naive)

vs Baseline:
  Naive RMSE:          XX.XX
  Seasonal Naive RMSE: XX.XX
  Melhora vs naive:    XX.X%

Diagnóstico:
  Autocorrelação resíduos: Sim / Não
  Viés sistemático:        Sim / Não
```

---

## Erros comuns

- ❌ Reportar MAPE quando série tem zeros — resultado enganoso
- ❌ Calcular métricas na escala transformada (log/diff) — reverta primeiro
- ❌ CV sem gap — métricas otimistas (vaza informação futura)
- ❌ Não calcular baseline — sem referência de comparação
- ✅ Sempre reportar desvio padrão do CV — RMSE=10 ± 8 é muito diferente de RMSE=10 ± 1
