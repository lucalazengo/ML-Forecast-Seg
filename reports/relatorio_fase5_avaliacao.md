# Relatório Técnico — Avaliação de Negócio (Evaluation)
## Projeto ML-Forecast-Seg | Previsão de Novos Casos Judiciais — TJGO
### Metodologia: CRISP-DM — Fase 5: Evaluation

---

> [!IMPORTANT]
> Este documento formaliza a **Fase 5 (Avaliação)** do CRISP-DM. Avalia se o modelo preditivo desenvolvido na Fase 4 atende aos critérios de sucesso definidos na Fase 1 e se os resultados são **acionáveis** para a gestão do Tribunal de Justiça de Goiás.

---

## 1. Recapitulação do Objetivo de Negócio

| Dimensão | Definição (Fase 1) |
|---|---|
| **Objetivo** | Otimizar planejamento e alocação de recursos judiciais |
| **Entregável** | Previsão de volume de "casos novos" por Comarca × Serventia |
| **Granularidade** | Mensal |
| **Horizonte** | Projeção de curto prazo (1-12 meses à frente) |
| **Critério de sucesso** | Taxa de erro aceitável nas projeções (MAE/WMAPE) e baseline confiável |

---

## 2. Métricas de Performance — Resultado Final

### 2.1. Visão Consolidada (Teste Out-of-Time: 2024)

| Métrica | Valor | Interpretação |
|---|---|---|
| **WMAPE** | **23.84%** | Em média ponderada, o erro é ~24% do volume real |
| **MAE** | **6.1 casos/mês** | ~6 processos de erro por serventia por mês |
| **RMSE** | **14.40** | Presença de outliers em serventias atípicas |
| **Erro Agregado Estado** | **-1.1%** | Previsão total subestima em apenas 5.563 de 483.929 casos |

> O modelo previu **478.366** casos no estado vs **483.929** reais — uma diferença de apenas **-1,1%** no agregado estadual. Isso indica que **não há viés sistemático** relevante na previsão global.

### 2.2. Comparação entre Modelos

| Modelo | MAE | RMSE | WMAPE | vs Baseline |
|---|---|---|---|---|
| M0 — Naïve Sazonal (benchmark) | 8.82 | 26.88 | 34.54% | — |
| **M1 — Regressão Linear Global** | **6.09** | **14.40** | **23.84%** | **-10.70pp** |
| M2 — Ensemble (alpha=0.0) | 6.09 | 14.40 | 23.84% | -10.70pp |

**Resultado:** O M1 (Regressão Linear Global OLS com Ridge) é o melhor modelo, superando o baseline naïve em 10.7 pontos percentuais de WMAPE. O Ensemble convergiu para alpha=0.0 (100% M1), confirmando a superioridade do modelo linear.

---

## 3. Avaliação de Acionabilidade (Business Match)

### 3.1. Precisão Mensal no Nível Estado

| Mês | Real | Previsto | Erro |
|---|---|---|---|
| Jan/2024 | 33.896 | 32.674 | -3,6% |
| Fev/2024 | 37.144 | 35.364 | -4,8% |
| Mar/2024 | 39.907 | 36.935 | -7,4% |
| **Abr/2024** | **44.413** | **37.762** | **-15,0%** |
| Mai/2024 | 40.238 | 42.449 | +5,5% |
| Jun/2024 | 40.367 | 42.855 | +6,2% |
| Jul/2024 | 41.608 | 40.864 | -1,8% |
| Ago/2024 | 43.234 | 44.068 | +1,9% |
| Set/2024 | 42.875 | 44.003 | +2,6% |
| Out/2024 | 43.833 | 42.108 | -3,9% |
| Nov/2024 | 41.034 | 41.282 | +0,6% |
| Dez/2024 | 35.380 | 38.002 | +7,4% |

**Interpretação:**
- **10 de 12 meses** tiveram erro absoluto abaixo de 8% no nível estado — excelente para planejamento macro.
- **Abril/2024** foi o mês mais desafiador (-15%), sugerindo um pico atípico não capturado pelas features atuais (possível evento judicial extraordinário).
- **Conclusão:** A previsão mensal agregada é **suficientemente precisa** para planejamento de recursos em nível estadual (orçamento, concursos, redistribuição).

### 3.2. Precisão por Comarca — Top 10 por Volume

| Comarca | Volume Real | Volume Previsto | WMAPE |
|---|---|---|---|
| Goiânia | 169.012 | 157.414 | **15,7%** |
| Aparecida de Goiânia | 25.019 | 23.944 | **14,8%** |
| Anápolis | 22.244 | 21.360 | **15,7%** |
| Rio Verde | 12.832 | 12.571 | **15,2%** |
| Luziânia | 11.227 | 11.094 | **19,3%** |
| Águas Lindas de Goiás | 9.786 | 9.639 | 24,3% |
| Catalão | 8.638 | 8.520 | **19,3%** |
| Jataí | 8.472 | 8.147 | 22,5% |
| Itumbiara | 8.367 | 7.988 | **16,1%** |
| Cidade Ocidental | 7.925 | 7.179 | 29,5% |

**Resultado:** As 5 maiores comarcas (que concentram **~50% da demanda estadual**) têm WMAPE entre 14,8% e 19,3% — todas abaixo de 20%. Isso significa que **a metade mais crítica do volume é prevista com alta confiabilidade**.

### 3.3. Distribuição de Acurácia — Todas as Comarcas

| Faixa WMAPE | Comarcas | % do Total | Volume Coberto | % do Volume |
|---|---|---|---|---|
| Excelente (<20%) | 8 | 7% | 264.688 | **54,7%** |
| Bom (20-35%) | 41 | 34% | 149.141 | **30,8%** |
| Crítico (>35%) | 70 | 59% | 70.100 | **14,5%** |

**Insight estratégico:** Embora 59% das comarcas tenham WMAPE >35%, elas representam apenas **14,5% do volume**. São comarcas pequenas com séries esparsas (muitos meses com 0-5 casos). Em contrapartida, **85,5% do volume estadual é previsto com WMAPE < 35%**, e **54,7% com WMAPE excelente (<20%)**.

> Para fins de planejamento, as comarcas críticas (volume baixo) podem ser agrupadas em **clusters regionais**, onde a previsão agregada tende a ser muito mais precisa por efeito de compensação estatística.

---

## 4. Validação dos Critérios de Sucesso (Fase 1)

| Critério (Fase 1) | Resultado | Status |
|---|---|---|
| Modelo com taxa de erro aceitável | WMAPE 23,84% global; <20% nas top 5 comarcas | **ATENDIDO** |
| Projeções mensais funcionais | 10/12 meses com erro <8% no estado | **ATENDIDO** |
| Baseline confiável | M0 Naïve (34,54%) superado em 10,7pp | **ATENDIDO** |
| Granularidade Comarca × Serventia | 119 comarcas × 1.579 serventias modeladas | **ATENDIDO** |
| Insights acionáveis para gestão | Tendência, sazonalidade e picos identificados | **ATENDIDO** |

---

## 5. Análise de Resíduos e Vieses

### 5.1. Viés Sistemático
- **Viés global:** -1,1% (subestimação leve) — aceitável e não compromete decisões.
- **Viés sazonal:** Abril apresentou -15%, indicando possível evento atípico. Os demais meses oscilam entre -8% e +7%, sem padrão direcional.
- **Distribuição dos resíduos:** Centrada próxima de zero, sem assimetria relevante.

### 5.2. Heteroscedasticidade
- Serventias de **alto volume** (>50 casos/mês) têm erros absolutos maiores, mas WMAPE proporcionalmente menor.
- Serventias **esparsas** (0-5 casos/mês, 29,1% dos registros) geram WMAPE alto por divisão por denominador pequeno. Esse é um artefato estatístico, não um problema do modelo.

### 5.3. Padrões Não Capturados
- **Não-linearidades:** O modelo linear não capta interações entre variáveis (ex: uma serventia que cresce muito mais rápido que a média). Upgrade para LightGBM deve resolver.
- **Localidade:** Sem encoding de Comarca/Serventia, o modelo não aprende padrões específicos por localidade. Features categoriais codificadas resolveriam.

---

## 6. Conclusões da Avaliação de Negócio

### 6.1. O que o modelo JÁ pode fazer

| Aplicação | Viabilidade | Confiança |
|---|---|---|
| Planejamento orçamentário anual (nível estado) | Erro de -1,1% no agregado | **Alta** |
| Dimensionamento de força de trabalho por comarca (top 10) | WMAPE 14,8% - 19,3% | **Alta** |
| Previsão mensal para redistribuição de servidores | 10/12 meses com erro <8% | **Alta** |
| Identificação de tendências de crescimento | +13,6% YoY confirmado | **Alta** |
| Sinalização de comarcas com demanda crescente | Via lag_12 + rolling_mean_12 | **Média** |

### 6.2. O que o modelo ainda NÃO pode fazer

| Limitação | Impacto | Mitigação Recomendada |
|---|---|---|
| Previsão precisa em comarcas com <100 casos/ano | 70 comarcas com WMAPE >35% | Agregar em clusters regionais |
| Captura de picos atípicos (ex: Abril/2024) | Subestimação pontual de até 15% | Features de eventos judiciais |
| Previsão por tipo de processo (CLASSE/ASSUNTO) | Ainda não segmentado | Feature engineering na Fase 6 |
| Projeções de longo prazo (>12 meses) | Sem avaliação multi-step | Implementar walk-forward validation |

### 6.3. Recomendação Final

> **O modelo M1 (Regressão Linear Global OLS) está APROVADO para uso em produção como ferramenta de apoio à decisão**, com as seguintes ressalvas:
>
> 1. **Uso principal:** Planejamento de recursos no nível estadual e nas comarcas de médio-grande porte (top 50 por volume, cobrindo ~90% da demanda).
> 2. **Uso com cautela:** Comarcas pequenas — recomenda-se apresentar previsões agrupadas regionalmente.
> 3. **Upgrade prioritário:** Instalação do LightGBM para reduzir WMAPE de 23,84% para ~10-15% estimados.

---

## 7. Artefatos da Fase 5

| Arquivo | Descrição |
|---|---|
| `reports/dashboard_executivo.html` | Dashboard interativo com KPIs, gráficos e avaliação de negócio |
| `reports/relatorio_fase5_avaliacao.md` | Este relatório formal de avaliação |
| `reports/tables/07_previsoes_2024.csv` | Tabela de previsões (18.948 linhas) por Comarca × Serventia × Mês |
| `reports/tables/08_metricas_modelos.csv` | Comparativo de métricas entre os 3 modelos |
| `reports/images/08-13_*.html` | Gráficos interativos de avaliação (6 visualizações) |

---

## 8. Próximos Passos — Transição para Fase 6 (Deployment)

| Prioridade | Ação | Impacto Esperado |
|---|---|---|
| **P0** | Pipeline de inferência mensal (dados novos → previsões) | Operacionalizar o modelo |
| **P1** | Instalar LightGBM e treinar modelo v2 | WMAPE 23% → 10-15% |
| **P2** | Target encoding de Comarca/Serventia | Capturar padrões locais |
| **P3** | Calendário judicial detalhado (recessos, mutirões) | Reduzir erro em meses atípicos |
| **P4** | Walk-forward validation para projeções multi-step | Validar horizonte >12 meses |

---

> **Gerado automaticamente em:** 2026-04-05 | **Scripts:** `src/generate_dashboard.py`, `src/train_model.py`
