# Plano de Implementação - Previsão de Casos (TJ)
**Metodologia:** CRISP-DM (Cross-Industry Standard Process for Data Mining)

## FASE 1: Business Understanding (Compreensão do Negócio)
*   **Objetivo do Negócio:** Otimizar o planejamento e a alocação de recursos judiciais, prevendo o volume de "casos novos" que darão entrada no tribunal.
*   **Objetivos de Data Science:** Desenvolver um modelo preditivo de Séries Temporais capaz de estimar o volume de novos processos de forma granular e segmentada, especificamente nos níveis territoriais e de competência: **Comarca** e **Serventia**.
*   **Critérios de Sucesso:** Modelos com taxa de erro aceitável nas projeções mensais/semanais (definir MAE/WMAPE alvo), e criação de baseline confiável.

## FASE 2: Data Understanding (Compreensão dos Dados)
*   **Fontes de Dados:** Diretório `data/raw/` contendo arquivos mensais/anuais em formato `.csv` de 2014 até 2024.
*   **Atributos Chave Identificados:**
    *   `NUMERO`: Chave primária (Contagem resultará na volumetria).
    *   `DATA_RECEBIMENTO`: Eixo temporal (Timestamp das séries temporais).
    *   `COMARCA`: Agrupador hierárquico 1.
    *   `SERVENTIA`: Agrupador hierárquico 2.
    *   `CLASSE`, `ASSUNTOS`, `AREA`: Potenciais variáveis exógenas de regressão ou dimensionais de segmentação profunda futura.
*   **Tarefas Atuais:** Análise descritiva (EDA), validação de tipos de dados (Datas válidas?), verificação de campos nulos e identificação da esparsidade nas Serventias.

## FASE 3: Data Preparation (Preparação dos Dados)
*   **Consolidação:** Ingestão iterativa dos CSVs de 2014 a 2024, padronizando os layouts.
*   **Agregamento (Resampling):** Agrupar a volumetria diária/semanal/mensal `COUNT(NUMERO)` pelas chaves temporais (`DATA_RECEBIMENTO`), `COMARCA` e `SERVENTIA`.
*   **Tratamento de Anomalias:** Imputação para períodos faltantes em Serventias (zeros onde não houve demanda) e tratamento de *outliers* (ex: meses focados em força-tarefa ou greves, ou suspensão de prazos pela pandemia em 2020/2021).
*   **Engenharia de Features:** 
    *   *Features Temporais:* Dia da semana, mês, feriados específicos nacionais e do judiciário (recesso forense).
    *   *Lags:* Quantidade de novos casos em t-1, t-2, t-12.

## FASE 4: Modeling (Modelagem)
*   **Estratégias de Modelagem Segmentada:** Dada a alta dimensionalidade (Muitas Comarcas x Serventias), escolher entre:
    *   *Abordagem Local:* Um modelo `ARIMA`, `Prophet` ou `ETS` individual para cada par Comarca-Serventia.
    *   *Abordagem Global Machine Learning:* Um modelo único de *gradient boosting* (ex: `LightGBM` ou `XGBoost`) ou Deep Learning (`DeepAR`, `TFT`) englobando todas as localidades (onde a Comarca/Serventia se torna uma *Feature Categorical*).
*   **Divisão Treino-Teste:** Definir o split via *Time Series Cross-Validation* ou separação *Out-of-Time* (ex: treinar em 2014-2023, validar em 2024).

## FASE 5: Evaluation (Avaliação) ✅ CONCLUÍDA

*   **Métricas de Performance:** WMAPE Global = **23,84%** | MAE = **6,1 casos/mês** | RMSE = 14,40. Modelo M1 (OLS Ridge) superou baseline naïve em **-10,7pp**.
*   **Validação Agregada:** Erro de apenas **-1,1%** no total estadual (478K previstos vs 484K reais). 10 de 12 meses com erro <8%.
*   **Acurácia por Comarca:** Top 5 comarcas (50% do volume) com WMAPE entre 14,8%-19,3%. 85,5% do volume estadual previsto com WMAPE <35%.
*   **Interpretabilidade e *Business Match*:** Modelo aprovado para planejamento de recursos em nível estadual e comarcas de médio-grande porte. Comarcas pequenas (59% das comarcas, mas apenas 14,5% do volume) requerem agregação regional.
*   **Dashboard Executivo:** `reports/dashboard_executivo.html` — KPIs, gráficos interativos e avaliação de negócio.
*   **Relatório Formal:** `reports/relatorio_fase5_avaliacao.md`

## FASE 6: Deployment (Implantação)
*   **Pipeline de Previsões:** Estruturar a inferência (mensal/diária) onde os dados mais recentes de carga retroalimentam o modelo e geram inferência para $t+n$.
*   **Disponibilização:** Geração de tabelas de previsões finais no banco de dados e/ou visualização via dashboard para apoiar as decisões estratégicas do Tribunal.
