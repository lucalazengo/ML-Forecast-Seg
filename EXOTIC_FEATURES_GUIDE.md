# 🔮 Engenharia de Features Exótica para WMAPE < 27%

## Problema Atual
- **M3 (LightGBM baseline)**: WMAPE **27.43%**, RMSE **67.79** ❌
- **M1 (Linear OLS)**: WMAPE **23.84%**, RMSE **14.4** ✅

**Insight**: O modelo linear é mais estável. LightGBM está gerando previsões extremas (RMSE muito alto). Precisamos de features que **regularizem** e **capturem padrões lineares** melhor.

---

## Técnicas Implementadas

### 1. 🌊 **Fourier Features** (8 features)
```
fourier_12m_sin_1, fourier_12m_cos_1
fourier_12m_sin_2, fourier_12m_cos_2
fourier_6m_sin_1, fourier_6m_cos_1
... (períodos: 12, 6, 4, 3 meses, ordens: 1-2)
```

**Por que funciona:**
- Captura múltiplos ciclos sazonais do sistema judiciário
- 12m: ciclo anual (períodos de férias forense)
- 6m: semestres judiciais
- 4m, 3m: variações intra-semestrais
- Mais suave que dummies, evita overfitting

**Esperado:** ↓ RMSE (menos picos), ~+1-2% WMAPE

---

### 2. 📊 **Quantile Features** (13 features)
```
rolling_q25_3, rolling_q50_3, rolling_q75_3 (janelas 3, 6, 12m)
rolling_iqr_3, rolling_iqr_6, rolling_iqr_12 (variabilidade)
```

**Por que funciona:**
- Captura comportamento em diferentes percentis (não só média)
- IQR (Interquartile Range) = proxy de volatilidade
- Detecta períodos de alta/baixa demanda sem assumir gaussiana
- Regulariza outliers

**Esperado:** ↓ WMAPE (melhor robustez), ↓ RMSE

---

### 3. 📈 **Detrended Features** (9 features)
```
detrended_3, detrended_6, detrended_12
trend_3, trend_6, trend_12
deviation_3, deviation_6, deviation_12
```

**Por que funciona:**
- Remove trend local (remove drift sistemático)
- Resíduos vs trend = anomalias
- Magnitude de desvios (deviation) = early warning
- O LightGBM consegue aprender quando algo tá "fora do normal"

**Esperado:** ↑ Acurácia em períodos normais, detecção de mudanças

---

### 4. 🎯 **Holt's Local Level/Slope** (2 features)
```
holt_level (nível subjacente suavizado)
holt_slope (tendência local)
```

**Por que funciona:**
- Suavização exponencial dupla = captura dinamicamente a tendência
- Sem assumir linearidade global
- Modela série como: `y[t] = level[t] + slope[t]*t + erro`
- Mais elegante que polinômios

**Esperado:** ↑ Captura de trends variáveis, melhor para OOT

---

### 5. 🔄 **Multiplicative Seasonal Indices** (2 features)
```
seasonal_index (força da sazonalidade: valor / média)
seasonal_index_log (versão log, mais estável)
```

**Por que funciona:**
- Diferente de dummies: captura **força** da sazonalidade
- Setembro pode ser 1.2× (20% mais casos)
- Janeiro pode ser 0.8× (20% menos casos)
- Mais informativo que `is_recesso=1`

**Esperado:** ↑ Flexibilidade sazonal, melhor interpretação

---

### 6. ⚡ **Volatility Dynamics** (5 features)
```
rolling_std_3_lag_1, rolling_std_3_lag_2, rolling_std_3_lag_3, rolling_std_3_lag_6
volatility_accel (derivada da volatilidade)
```

**Por que funciona:**
- Lags da volatilidade = momentum da incerteza
- Se std tá crescendo → períodos turbulentos chegando
- Aceleração = mudança abrupta (court closures, legal changes)
- LightGBM pode usar como "feature de confiança"

**Esperado:** ↓ WMAPE em períodos de transição

---

### 7. 🚀 **Rate-of-Change Features** (8 features)
```
roc_1, roc_3, roc_6, roc_12 (absoluto)
roc_pct_1, roc_pct_3, roc_pct_6, roc_pct_12 (percentual)
```

**Por que funciona:**
- Captura momentum (crescimento/declínio)
- roc_pct_1 = mudança mês-a-mês (volatilidade curta)
- roc_pct_12 = crescimento anual (estrutural)
- Early warning de mudanças abruptas

**Esperado:** ↑ Detecção de regime changes

---

### 8. 🌍 **Cross-Sectional Features** (4 features)
```
comarca_mean (efeito regional)
comarca_mean_lag (região tendência)
deviation_from_comarca (outlier relativo)
serventia_comarca_ratio (normalização inter-regional)
```

**Por que funciona:**
- Captura efeitos regionais latentes
- Goiânia ≠ Anápolis (tamanhos diferentes)
- Serventia pode ser "melhor/pior" que sua comarca
- Regulariza extrapolação para novas serventias

**Esperado:** ↑ Generalização cross-sectional

---

### 9. 🚨 **Anomaly Features** (2 features)
```
is_anomaly (z-score > 2.5σ)
anomaly_count_6m (contagem de anomalias recentes)
```

**Por que funciona:**
- Sinaliza períodos atípicos (court closures, pandemias)
- LightGBM pode aprender "quando não confiar em lags"
- Evita extrapolação de outliers

**Esperado:** ↓ WMAPE em períodos normais (menos efeito de outliers)

---

## Total de Novas Features
- **8** Fourier features
- **13** Quantile features
- **9** Detrended features
- **2** Holt level/slope
- **2** Seasonal indices
- **5** Volatility features
- **8** Rate-of-change features
- **4** Cross-sectional features
- **2** Anomaly features

**Total: 53 novas features** (+ 15 originais = **68 features**)

---

## Como Usar

### Opção 1: Aplicar e Retreinar (Recomendado)

```bash
cd src
python enhance_with_exotic_features.py
```

Isso vai:
1. ✅ Carregar dados preparados
2. ✅ Aplicar todas as 53 features
3. ✅ Retreinar LightGBM com hiperparâmetros tuned
4. ✅ Exportar resultados e comparar WMAPE

**Tempo esperado**: 5-15 min (depende de CPU)

### Opção 2: Aplicar em Novo Dataset Apenas (Para Previsões 2026)

```python
from exotic_features import apply_all_exotic_features

# Seu df com dados históricos
df_com_features, feature_names = apply_all_exotic_features(df)
```

---

## Expectativas de Melhoria

### Cenário Otimista 🎯
- **M4 WMAPE: 21-23%** (↓ 4-6%)
- RMSE: 12-14
- MAE: 4-5

### Cenário Realista ✅
- **M4 WMAPE: 23-25%** (↓ 2-4%)
- RMSE: 14-16
- MAE: 5-6

### Cenário Conservador ⚠️
- **M4 WMAPE: 25-27%** (↓ 0-2%)
- Ainda assim ajuda LightGBM a competir com OLS

---

## Diagnóstico: Por que LightGBM tá pior que OLS?

| Métrica | OLS | LightGBM | Razão |
|---------|-----|----------|-------|
| WMAPE | 23.84% | 27.43% | LGB overfitting |
| RMSE | 14.4 | 67.79 | Outliers extremos |
| MAE | 6.09 | 14.73 | Previsões errantes |

**Diagnóstico:**
- LightGBM consegue aprender interações, mas tá sendo "muito criativo"
- Sem features que expliquem comportamento linearmente, ele inventa
- Exotic features = informação estruturada = menos overfitting

---

## Tuning Adicional (Se necessário)

Se WMAPE não melhorar suficiente, tente:

### A. Aumentar Regularização
```python
reg_alpha=1.0  # foi 0.5
reg_lambda=2.0  # foi 1.0
learning_rate=0.03  # foi 0.04
```

### B. Reduzir Complexidade
```python
max_depth=8   # foi 10
num_leaves=63  # foi 95
```

### C. Feature Selection
Se modelo tá lento, remova features menos importantes:
```python
# Pegar top 40 features por importância
top_features = importances_df.head(40)['Feature'].tolist()
```

### D. Ensemble LGB + Linear
```python
pred_final = 0.6 * y_pred_lgb + 0.4 * y_pred_ols
```

---

## Interpretabilidade

Depois de retreinar, veja:

```bash
# Top 15 features
cat reports/tables/10_feature_importance_exotic.csv | head -15
```

Espera-se ver:
- Fourier features (sazonalidade)
- Lags originais (memória)
- Detrended features (anomalias)
- Cross-sectional (efeitos regionais)

---

## Próximos Passos

1. ✅ Executar `enhance_with_exotic_features.py`
2. ✅ Comparar WMAPE: M4 vs M3 vs M1
3. ✅ Se WMAPE < 25%, usar M4 para previsões 2026
4. ✅ Senão, tentar ensemble ou feature selection

---

## Referências Técnicas

- **Fourier Features**: Campbell & Walker (2007) - Seasonal Adjustment
- **Quantile Regression**: Koenker & Bassett (1978)
- **Detrending**: Hodrick & Prescott (1997)
- **Holt-Winters**: Holt (2004), Winters (1960)
- **Anomaly Detection**: Tukey's Fences (3× IQR rule)

---

**Criado:** 2026-04-08 | **Versão:** Exotic Features v1.0
