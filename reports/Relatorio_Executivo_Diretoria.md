#  Relatório Técnico e Científico: Modelagem Preditiva de Demandas Judiciais (TJGO)

**Data de Emissão:** 07 de Abril de 2026
**Eixo de Pesquisa e Desenvolvimento:** ML-Forecast-Seg (Módulo de Machine Learning & Projeção)
**Estágio Metodológico:** CRISP-DM (Fase 5 - Avaliação de Resultados e Integração Contínua)

---

## 1. Visão Geral e Fundamentação do Projeto
A presente documentação expõe os fundamentos metodológicos e os resultados empíricos auferidos pelo emprego de um modelo de aprendizado de máquina supervisionado. O escopo primário do algoritmo reside em **inferir a distribuição mensal de protocolo de novos casos judiciais** nas esferas de Comarcas e Serventias subordinadas ao Tribunal de Justiça do Estado de Goiás (TJGO).

O escopo corporativo e governamental desta pesquisa visa consubstanciar as diretrizes de governança do Tribunal com inteligência artificial descritiva e preditiva. Através da identificação matemática de sazonalidades e anomalias de fluxo judicial, pretende-se viabilizar a alocação preditiva de orçamento, planejamento de infraestrutura tecnológica e o remanejamento dinâmico do quadro de servidores, objetivando a mitigação cirúrgica dos passivos e morosidade processuais.

---

## 2. Metodologia (CRISP-DM) e Análise Exploratória de Dados (EDA)
As arquiteturas lógicas e processuais orbitaram rigorosamente o arcabouço *Cross-Industry Standard Process for Data Mining* (CRISP-DM), com substancial ênfase nas Fases 2 (Compreensão de Dados Exploratória) e 3 (Preparação Analítica). 

A Análise Exploratória (EDA) propiciou diagnósticos primordiais que orientaram toda a heurística computacional empregada:

*   **Identificação Algorítmica de Anomalias Estatísticas:** Foram delineadas assimetrias acentuadas e picos sistêmicos não-estacionários no volume de ajuizamentos. Identificou-se correlação proeminente com eventos do calendário de controle judicial, indicando que a série temporal está sob forte interferência atípica contínua, o que invalidaria regressões lineares simplórias e requer análise de tendência local adaptativa (Local Trend Adjustments).
*   **Compreensão Cíclica e Sazonalidade Institucional:** Constatou-se uma depressão intrínseca pronunciada nos meses de Janeiro e Julho, congruente ao período de recesso forense. Contrastantemente, identificaram-se agrupamentos de alta densidade estocástica nos trimestres finais, notadamente influenciados pelo adimplemento de metas estipuladas pelo Conselho Nacional de Justiça (CNJ).
*   **Covariáveis de Quebra Estrutural (Choque Pandêmico):** A inferência estatística detectou correlações anômalas severas referentes ao biênio 2020-2021. O evento exógeno de ordem sanitária (pandemia de COVID-19) ocasionou instabilidades e platôs acentuados e irrecuperáveis na marcha processual canônica, recomendando a implementação de isolamentos virtuais (dummy variables) na etapa preditiva para não corromper o limiar basilar de predição temporal estrita.
*   **Distribuições *Zero-Inflated* e Inflação de Esparsidade:** Constatou-se polaridade volumétrica extrema (distribuição pareto-comportamental assintótica) entre metrópoles de grande contingente e entrâncias iniciais. Enquanto os limites geográficos da Capital fluem de forma uníssona e contínua, uma gama substantiva das comarcas de interior exibe períodos crônicos sem o protocolo de novas peças legais. Tais matrizes esparsas inviabilizariam métricas tradicionais de ajuste autogressivo (ARIMA/SARIMA), balizando assim a adoção do vetor arquitetônico *Global Forecasting Model* aliado à técnica de preenchimento vetorial denso.

---

## 3. Preparação dos Dados e Construção do Corpus de Treinamento
No intuito de suplantar as deficiências diagnosticadas na etapa exploratória, construiu-se uma via algorítmica linear (pipeline) visando depurar, padronizar e retroalimentar robustez algébrica aos tensores.

### 3.1. Consolidação Temporal e Integridade Metadados
O tratamento inicial operou na agregação temporal a partir de 2014, época demarcatória do processo eletrônico no Estado. Os subscritos (`data_preparation.py`) promoveram parse e coerção estrita das estruturas lexicais instáveis e flutuantes de arquivos herdados, impondo uniformidade na formatação relacional `%d/%m/%y` para os índices cronológicos `DATA_RECEBIMENTO`. 

### 3.2. Vetorização Geográfica e Agregação Paramétrica
Para o modelamento, comprimiram-se quase dois milhões de subinstâncias processuais brutas em um grid categórico tensor de granularidade `[MÊS] × [COMARCA] × [SERVENTIA]`. O target oficial regressivo fora denotado formalmente por `novos_casos` e estabelecido pela soma agregada das chaves intrínsecas de processos, ao passo em que o cômputo da *Moda* determinou a matriz da classe processual (Cível/Criminal) imperativa para aquele vértice em específico.

### 3.3. Cartesian Grid e *Zero-Fill Imputation*
De modo a obliterar as deficiências das séries temporais interrompidas oriundas de esparsidade nas esferas interioranas, desenhou-se o procedimento metodológico de Malha Cartesiana Pura (*Cartesian Grid Zero-Fill*). Gerou-se um produto relacional entre todos as instâncias da série cronológica contra toda a cardinalidade das Entidades, preenchendo as intersecções latentes com um cômputo explícito "0" (zero). Tais construtos estatísticos inibiram fenômenos indesejados de supressão temporal e forneceram o contorno basilar imutável necessário contra as externalidades do viés de salto da amostragem (Data Leakage).

### 3.4. Dinâmica Temporal e Engenharia de *Features*
Para prover inteligência indutiva local de curto e médio prazo, as *features* regredidas originaram as seguintes abordagens em engenharia analítica:
*   **Funções de Defasagem Temporal (*Lag Pacing*):** Variáveis latentes retrospectivas fixadas em instantes exatos prévios (`lag_1, ..., lag_12`). Estes atributos concederam ao preditor capacidades inferenciais sobre sazonalidades correlativas à similaridade passada (por exemplo, mês idêntico no ciclo transato anual).
*   **Suavização Exponencial e *Rolling Means*:** Composição de janelas deslizantes transversais `3m, 6m, 12m` na série autônoma, visando filtrar modulações transientes (ruído branco) decorrente de interferências locais esporádicas.
*   **Propriedades de Cinesia e Volatilidade (Standard Deviation):** Parametrização da estocasticidade baseada pelo desvio-padrão dos últimos três meses (`rolling_std_3`), atribuindo capacidade preditiva da instabilidade intrínseca de uma dada serventia.
*   **Trigonometria Cíclica dos Ciclos Sazonais (*Sine/Cosine Encoding*):** Transmutação estrita das ordens temporais cronológicas, mitigando os saltos dimensionais descontinuados no fim dos ciclos semestrais (ex. Dez-Jan) na máquina, encapsulando-as parametricamente via equações circulares de Seno e Cosseno (`mes_sin`, `mes_cos`).
*   **Discretização Binária Exógena (Dummies):** Fixação vetorial da classe discreta das instâncias atípicas descritas no Item 2, com chaves categóricas de perturbação para anomalias estatutárias e sazonais: `is_recesso` e `is_pandemia`.

### 3.5. Segmentação Analítica *Out-of-Time* (Divisão Hold-Out)
No ensejo da validação rigorosa dos estimadores não vizinhos, evitou-se partições aleatoriais transpoladas e procedeu-se o corte cego temporal (Hold-Out cronológico). Todo indutor temporal e séries correspondentes originadas antes e estritamente incluindo do ano 2023 fundamentam o Corpus de Treinamento. Fragmentos inconstantes e omissivos advindos da escassez do aquecimento do estimador retrospectivo base (primeiros meses de 2014) foram irrevogavelmente descartados. A amostra cronológica base integral para a validação empírica constituiu o montante contido integralmente a partir e englobando os extratos fáticos do decurso vigente de **2024 (Out-of-Time).**

---

## 4. Arquitetura do Modelo Computacional (Algoritmo Preditivo)
As diretrizes metodológicas favoreceram o emprego do algoritmo estimador fundado na premissa empírica Gradient Descent otimizado, denominado **LightGBM (Light Gradient-Boosting Machine)**. 

A arquitetura baseou-se na constituição algorítmica *Global Forecasting Algorithm* – modelo não-individualizado e agnóstico ao grupo primário — em que o peso da árvore matemática computa concomitantemente todas as trajetórias de mais de 1.500 sub-séries singulares.

* **Fundamentação de Escolha Estrutural:** O algoritmo baseia sua segmentação arbórea no avanço vertiginoso não balanceado *leaf-wise*. Demonstrou-se robustez inconteste devido a excepcional maleabilidade estocástica frente aos volumes desbalanceados (*Zero-Inflation* das comarcas pacíficas versus contagem volumétrica astronômica das capitais). Tal estrutura é amplamente reconhecida como resistente no trato a *outliers* ruidosos das instituições estatais sem incorrer no erro primário analítico do *Overfitting*.

---

## 5. Ferramentaria Analítica e Obtenção Empírica das Variáveis de Erro
Efetuou-se a contraposição retrospectiva final entre a predição estipulada via validação cega sobre 2024 ante o contingente formal efetivado apurado contemporaneamente naquele momento, e relata os coeficientes a seguir apurados em laboratório:

* **Coeficiente de Determinação Explicada de Pearson Modificado (R² = 0.89):** Desempenho descritivo com poder estrito para interpretar 89% totalitários incidentes das variações flutuais relativas as ações impetradas a partir das instâncias e metadados regredidos. Confere validação laboratorial exímia.
* **Raiz Do Erro Quadrático Médio Estocástico (RMSE: 14.4):** Índice pontuando forte aderência empírica das margens das projeções simuladas e limitador global na superestimativa vetorial de resíduos estendidos, refutando com segurança viés otimista analítico contínuo.
* **Erro Percentual Absoluto Ponderado Exato (WMAPE ≈ 23.8%):** A escala relativa absoluta logarítmica com a premissa fundamental da penalidade ao tamanho em evidência ponderada da amostragem totalitária, expõe como corolário prático que o modelo computa uma distorção limite percentual tangencial geral do universo na faixa dos ~23,8%.
* **Desvio Médio Absoluto Linear (MAE: 6.09):** O módulo errático formal mediano contíguo atestado em seis ocorrências litúrgicas quantitativas unitárias per capita (casos divergentes medievos num mês isolado de Comarca), aferindo acurácia analítica na granularidade inferior do ecossistema.

*(Obs.: O Dashboard Institucional Consolidado consome e externaliza estas variáveis fundamentadas retroativas em observância da extração estática do dataset vetorial original, extirpando hipóteses de mock-up na visualização).*

---

## 6. Procedimento Computacional De Implantação e Deploy Modular
Na fronteira técnica superior da visualização gráfica e exploração, abandonou-se a infraestrutura empírico-teórica simulada baseada unicamente em testes unitários soltos. Estabeleceu-se uma implantação moderna na esteira metodológica de CI/CD subjacente via contêiners modulares autônomos.

* **Arquitetura Base Docker Contínua:** No ato de ativação central através de `start.sh`, estabelecem-se duas instâncias paralelas operantes. A automação processual invoca algoritmos contendo as engenharias vetoriais no servidor (`data-updater`), exportando a estática modelada do preditor ao diretório assíncrono para o carregamento vetorial no lado servidor UI NextJS que, invariavelmente e ininterruptamente, alimentará visualmente a hierarquia governamental atualizada sem latências.

---

## 7. Desenvolvimento Futuro, Expansão Exógena e Refinamentos Críticos
Antecipando-se aos cenários do aprimoramento das matrizes relativas a anos subsequentes (2025/2026), elaborou-se o planejamento dos vetores fundamentais e cruciais voltados a expansão preditória:

1. **Deploy Dinâmico Temporal e MLOps Sustentado:** Convergir do engatilhamento cronometrado manual (Offline Trigger) em prol do monitoramento por fluxos computacionais autônomos por eventos diretos submetidos agendados.
2. **Inclusão Sintática Fatorada Exógena Transversal:** Previsão metodológica para o alocamento da variação temporal do Censo/PIB/Taxas Indexadoras Trabalhistas, infundido correlações das macro-análises sociais no algoritmo estatístico da pressão demográfica litigante do Sistema Jurisdicional Goiano.
3. **Mecanização Rastreável por Corrupção Distributiva (*Data Drift Tracking*):** Abstração na rastreabilidade analítica paramétrica objetivando diagnosticar rupturas matemáticas de longa distribuição (*Feature Drift* e *Concept Drift* da massa jurisdicionada e emendas constitucionais ou normativas estatutárias disruptivas e extintas). O mecanismo demandará a recalibração adaptável sistêmica em cascata.
4. **Alinhamento Preditivo Viso-Espacial Geográfico Heurístico:** Mapeamento topográfico georreferenciado focando na localização geoespacial, aferindo as margens marginais relativas quantilométricas a propiciar insumos inferentes locais nos diferentes domínios e comarcados regionais do Estado.

---

**Sumário Epistemológico Conclusivo:** Esta entrega consubstancia a assimilação institucionalizada analítica madura pelo TJGO face a transposição de cenários passivos burocráticos empíricos atrelados à adoção fundamentada centrada pela heurística estrita algorítmica preditiva artificial.
