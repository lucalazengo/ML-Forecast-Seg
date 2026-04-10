---
name: ts-outlier-anomaly
description: >
  Detecção e tratamento especialista de outliers e anomalias em séries temporais
  epidemiológicas no projeto ML-Forecast-Seg. Use esta skill quando: identificar
  picos suspeitos na série de casos_novos, tratar anomalias antes de feature
  engineering, classificar mudanças de nível vs spikes pontuais, ou adicionar
  features binárias de surtos ao dataset. Aplicar sempre após pré-processamento
  e antes de feature engineering.
---

# Skill: Outliers e Anomalias em Séries Temporais Epidemiológicas

## Script pronto

```bash
python .claude/skills/03_outlier_anomaly/scripts/outlier_detection.py
```

---

## Tipos de anomalia em dados de casos novos

| Tipo           | Comportamento visual        | Causa comum                    | Tratamento          |
|----------------|-----------------------------|--------------------------------|---------------------|
| **Spike**      | Pico pontual, volta ao nível | Surto real ou erro de notificação | Interpolar ou manter + flag |
| **Level shift**| Mudança de patamar persistente | Novo protocolo de notificação | Manter + feature binária |
| **Sazonalidade quebrada** | Padrão semanal desaparece | Feriados prolongados, greves | Feature de evento especial |
| **Zero run**   | Zeros consecutivos          | Falha no registro              | Forward fill curto |

---

## Pipeline de detecção (em ordem)

### Passo 1 — Z-score rolling (não global!)

```python
# Z-score GLOBAL é errado para séries temporais com tendência
# Z-score ROLLING captura contexto local
window = 30       # dias de contexto
threshold = 3.0   # padrão; abaixar para 2.5 se série muito ruidosa

rolling_mean = series.rolling(window, center=True, min_periods=7).mean()
rolling_std  = series.rolling(window, center=True, min_periods=7).std()
z_scores     = (series - rolling_mean) / (rolling_std + 1e-8)
mask_zscore  = z_scores.abs() > threshold
```

### Passo 2 — IQR rolling (robusto a assimetria)

```python
q1 = series.rolling(window, center=True, min_periods=7).quantile(0.25)
q3 = series.rolling(window, center=True, min_periods=7).quantile(0.75)
iqr = q3 - q1
mask_iqr = (series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)
```

### Passo 3 — Combinar detectores

```python
# Interseção: só flagra quando AMBOS concordam (menos falsos positivos)
combined = mask_zscore & mask_iqr

# União: flagra quando QUALQUER UM detecta (mais sensível)
combined = mask_zscore | mask_iqr
```

**Recomendação para epidemiologia:** usar **interseção** — falsos positivos
em dados de saúde têm custo alto (modelo perde informação real de surtos).

---

## Árvore de decisão pós-detecção

```
Outlier detectado?
    │
    ├─ Isolado (sem vizinhos anômalos)?
    │       └─ SPIKE
    │           ├─ Confirmado como surto real? → Manter + feature is_surto=1
    │           └─ Suspeito de erro?           → Interpolar
    │
    └─ Cluster (vizinhos também anômalos)?
            └─ LEVEL SHIFT
                └─ Manter valor + feature is_level_shift=1
                   (não interpolar — o novo patamar é real)
```

---

## Tratamento por tipo

```python
# SPIKE confirmado como erro → interpolar
df.loc[spike_mask, TARGET_COL] = np.nan
df[TARGET_COL] = df[TARGET_COL].interpolate(method='linear')

# SPIKE real (surto) → manter + adicionar flag
df['is_surto'] = 0
df.loc[surto_dates, 'is_surto'] = 1

# LEVEL SHIFT → adicionar feature binária de regime
df['regime'] = 0
df.loc[df.index >= data_mudanca, 'regime'] = 1
# Considerar treinar modelos separados por regime se shift for grande
```

---

## Quando treinar modelos separados por período

Se houver level shift grande (ex: mudança de protocolo de notificação):

```python
data_corte = '2022-03-01'  # data do evento
df_pre  = df[df.index < data_corte]
df_post = df[df.index >= data_corte]

# Treinar modelo separado para cada período
model_pre  = train_model(df_pre)
model_post = train_model(df_post)

# Usar apenas model_post para previsões futuras
```

---

## Checklist antes de prosseguir

- [ ] Visualizar série com outliers marcados (`plot_outliers()`)
- [ ] Classificar cada outlier (spike vs level shift)
- [ ] Documentar decisão de tratamento para cada outlier relevante
- [ ] Adicionar features binárias para eventos mantidos
- [ ] Confirmar que série limpa não tem NaN residuais
- [ ] Salvar série tratada em `data/processed/series_outlier_treated.parquet`

---

## Erros comuns

- ❌ Usar z-score global em série com tendência — classifica tendência como outlier
- ❌ Interpolar level shifts — apaga mudança real de patamar
- ❌ Threshold muito baixo (< 2.0) — muitos falsos positivos em dados epidemiológicos ruidosos
- ✅ Sempre plotar antes de tratar — inspeção visual é insubstituível
- ✅ Logar todas as datas com outlier detectado para rastreabilidade
