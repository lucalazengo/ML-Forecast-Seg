---
name: ts-feature-engineering
description: >
  Feature engineering especialista para XGBoost/LightGBM em séries temporais
  no projeto ML-Forecast-Seg. Use esta skill quando: criar lags e rolling stats,
  adicionar features de calendário, codificação cíclica de datas, features de
  feriados, tendência e momentum. É a skill de maior impacto na acurácia dos
  modelos — use sempre antes de qualquer treino.
---

# Skill: Feature Engineering para Forecast com XGBoost/LightGBM

## Por que esta skill é crítica

XGBoost e LightGBM **não têm memória temporal nativa**.
Toda informação sobre o passado da série precisa virar uma coluna explícita.
Feature engineering mal feita = modelo ruim, independente de hiperparâmetros.

## Script pronto

```bash
python .claude/skills/02_feature_engineering/scripts/build_features.py
```

Ou importe `build_all_features()` no seu pipeline.

---

## Regra de ouro: Nunca vazar o futuro

```
Horizonte = 7 dias → lag mínimo = 7
lag_1 com horizon=7 → LEAKAGE GARANTIDO ❌
lag_7 com horizon=7 → correto ✅
rolling_mean sem shift(horizon) → LEAKAGE ❌
rolling_mean com .shift(horizon) → correto ✅
```

**Verifique sempre:**
```python
# Checagem rápida de leakage
corr = df[f'lag_{lag}'].shift(-lag).corr(df[TARGET_COL])
# Se corr ≈ 1.0 → leakage. Se < 0.9 → provavelmente ok
```

---

## Catálogo de features por categoria

### 1. Lags (mais importantes)

```python
# Lags base — sempre incluir
for lag in [horizon, horizon+7, horizon+14, horizon+21]:
    df[f'lag_{lag}'] = df[TARGET_COL].shift(lag)
```

| Feature   | Captura                          |
|-----------|----------------------------------|
| `lag_7`   | Valor de exatamente 7 dias atrás |
| `lag_14`  | Padrão quinzenal                 |
| `lag_21`  | Memória de 3 semanas             |
| `lag_28`  | Sazonalidade mensal              |

### 2. Rolling statistics

```python
shifted = df[TARGET_COL].shift(horizon)  # shift obrigatório!

for w in [7, 14, 28]:
    df[f'rolling_mean_{w}']  = shifted.rolling(w).mean()
    df[f'rolling_std_{w}']   = shifted.rolling(w).std()
    df[f'rolling_min_{w}']   = shifted.rolling(w).min()
    df[f'rolling_max_{w}']   = shifted.rolling(w).max()
    # Volatilidade relativa
    df[f'rolling_cv_{w}']    = df[f'rolling_std_{w}'] / (df[f'rolling_mean_{w}'] + 1e-8)
```

### 3. Calendário com codificação cíclica

```python
# Codificação cíclica — CRÍTICO para dia_semana e mes
# Evita que o modelo veja 0 (seg) e 6 (dom) como distantes
df['dia_semana_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df['dia_semana_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
df['mes_sin']        = np.sin(2 * np.pi * (df.index.month - 1) / 12)
df['mes_cos']        = np.cos(2 * np.pi * (df.index.month - 1) / 12)
```

### 4. Tendência e momentum

```python
shifted = df[TARGET_COL].shift(horizon)
df['expanding_mean']  = shifted.expanding().mean()    # histórico acumulado
df['pct_change_7']    = shifted.pct_change(7)         # variação semanal
df['pct_change_28']   = shifted.pct_change(28)        # variação mensal
df['trend_index']     = np.arange(len(df))            # tendência linear bruta
```

---

## Estratégia multi-step (horizonte > 1)

### Direct (recomendado para horizonte ≤ 14 dias)
Um modelo por step — mais preciso, não acumula erro.

```python
models = {}
for h in range(1, HORIZON + 1):
    # Cria features com shift=h para cada horizonte
    df_h = build_all_features(df, horizon=h)
    X, y = df_h[feature_cols], df_h[TARGET_COL]
    models[h] = train_model(X, y)
```

### Recursive (mais simples, acumula erro)
Um único modelo, usa previsão anterior como input.
Aceitar apenas se horizonte ≤ 3 dias.

---

## Seleção de features

Após criar todas as features, use importância do modelo para filtrar:

```python
import pandas as pd

# Após treinar o modelo:
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

# Manter top-20 ou features com importância > 0.01
selected = importance[importance > 0.01].index.tolist()
print(f"Features selecionadas: {len(selected)}/{len(feature_cols)}")
```

---

## Erros comuns

- ❌ `dropna()` antes de criar features — perde contexto histórico
- ❌ Lags sem respeitar horizonte — leakage silencioso
- ❌ Usar `dayofweek` como inteiro sem codificação cíclica — modelo vê segunda (0) e domingo (6) como distantes
- ❌ Rolling sem `.shift(horizon)` — usa dados do futuro
- ✅ Sempre criar features em função do horizonte configurado em `config.py`
