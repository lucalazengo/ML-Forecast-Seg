---
name: ts-data-preprocessing
description: >
  Pré-processamento especialista de séries temporais para o projeto ML-Forecast-Seg.
  Use esta skill quando: carregar dados brutos de casos novos, detectar e corrigir
  gaps temporais, imputar valores faltantes, testar estacionariedade (ADF),
  aplicar transformações (log, diferenciação), ou qualquer tarefa de limpeza
  da série antes de feature engineering ou modelagem.
---

# Skill: Pré-processamento de Séries Temporais

## Script pronto

Use o script completo em `scripts/preprocess.py`.
Execute diretamente ou importe as funções no seu pipeline.

```bash
python .claude/skills/01_data_preprocessing/scripts/preprocess.py
```

---

## Checklist obrigatório — nessa ordem

- [ ] 1. Validar schema (colunas `data` e `casos_novos` presentes)
- [ ] 2. Ordenar por data e garantir frequência constante (`asfreq`)
- [ ] 3. Identificar e logar gaps na série
- [ ] 4. Imputar missing values com estratégia correta por tamanho do gap
- [ ] 5. Testar estacionariedade (ADF)
- [ ] 6. Aplicar transformação se necessário e registrar função inversa
- [ ] 7. Salvar em `data/processed/series_clean.parquet`

---

## Regras críticas de imputação

| Tamanho do gap | Estratégia         | Justificativa                        |
|----------------|--------------------|--------------------------------------|
| ≤ 3 períodos   | Interpolação linear | Variação gradual esperada            |
| 4–7 períodos   | Forward fill        | Mantém nível sem inventar tendência  |
| > 7 períodos   | **Investigar causa** | Pode ser mudança de protocolo        |

**NUNCA usar `fillna(mean)` ou `fillna(median)` em séries temporais.**
Isso destrói autocorrelação e cria picos artificiais.

---

## Teste de estacionariedade (ADF)

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(serie, autolag='AIC')
p_value = result[1]

# p < 0.05 → estacionária (pode modelar direto)
# p >= 0.05 → não estacionária → aplicar transformação
```

### Árvore de decisão pós-ADF

```
p >= 0.05 (não estacionária)?
    ├─ Série tem tendência suave? → log1p transform
    ├─ Série tem tendência forte? → diferenciação de ordem 1
    └─ Série tem sazonalidade forte? → diferenciação sazonal (shift=7 ou 52)

p < 0.05 (estacionária)?
    └─ Pode prosseguir sem transformação
```

### Reversão obrigatória na previsão

```python
# Se aplicou log1p:
previsao_original = np.expm1(previsao_transformada)

# Se aplicou diff:
# Precisa do último valor real observado
ultimo_valor_real = serie_original.iloc[-1]
previsao_original = previsao_transformada.cumsum() + ultimo_valor_real
```

---

## Diagnóstico visual rápido

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Série bruta
axes[0].plot(df[TARGET_COL])
axes[0].set_title('Série Original')

# Rolling mean e std (detecta não-estacionariedade visualmente)
rolling = df[TARGET_COL].rolling(window=28)
axes[1].plot(df[TARGET_COL], alpha=0.5, label='original')
axes[1].plot(rolling.mean(), label='média móvel 28d')
axes[1].plot(rolling.std(), label='std móvel 28d', linestyle='--')
axes[1].legend()
axes[1].set_title('Estacionariedade Visual')

# Distribuição
axes[2].hist(df[TARGET_COL].dropna(), bins=40)
axes[2].set_title('Distribuição dos Valores')

plt.tight_layout()
plt.savefig('reports/diagnostico_serie.png', dpi=150)
```

---

## Erros comuns a evitar

- ❌ `df.dropna()` antes de `asfreq()` — apaga contexto de gaps
- ❌ `sort_values` após `set_index` — use antes
- ❌ Imputar sem registrar onde houve gap — perde rastreabilidade
- ❌ Aplicar transformação e esquecer de reverter na previsão final
- ✅ Sempre salvar série pré-processada em `.parquet` (preserva dtypes e índice)
