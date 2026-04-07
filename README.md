#  ML-Forecast-Seg: Previsão de Casos Novos (TJGO)

Este repositório contém o modelo de aprendizado de máquina ponta a ponta (CRISP-DM) e o dashboard operacional concebidos para inferir a entrada de novos processos nas Comarcas e Serventias do Tribunal de Justiça do Estado de Goiás.

## Descrição da Solução
O **Módulo de Machine Learning & Projeção** atua consumindo dezenas de milhões de registros históricos desde 2014, construindo malhas de esparsidade matemática (*Cartesian Grid Zero-Fill*) para lidar com as comarcas sem processos intermitentes e mitigando as sazonalidades brutas inerentes aos recessos do Judiciário e afins através de algoritmos de Regressão Arbórea Não-Linear (**LightGBM**). 

A interface operacional é exposta atrávez de um contêiner **React/Next.js**.

##  Como Executar o Contêiner Completo

O projeto possui uma infraestrutura 100% conteinerizada (Docker Compose) que suprime a necessidade de configurações longas de ambientes Python/Node.

### Pré-requisitos
- **Docker** e **Docker Compose** instalados.

### Passos
1. Certifique-se de estar na pasta raiz do repositório (`ML-Forecast-Seg`).
2. Certifique-se que o script bash possui permissões execução:
   ```bash
   chmod +x start.sh
   ```
3. Inicie o agrupamento através do script principal:
   ```bash
   ./start.sh
   ```

### O que o `start.sh` faz?
- Sobe um container `data-updater` rodando **Python 3**.
- O container exporta toda a inferência e malha calculada pelo modelo para dentro das pastas visuais `/dashboard/public/data`.
- Finalizado o processamento quantitativo, ele levanta o Painel de Controle (Dashboard) em **Node/Nextjs** rodando a porta `3000`.

**Acesse no navegador:** [http://localhost:3000](http://localhost:3000)

##  Estrutura de Diretórios
* `src/`: Motores de extração lógica e Data Preparation.
* `dashboard/`: Código fonte NextJS que renderiza os gráficos e leitura JSON.
* `data/`: Extratos em lote, arquivos CSV processados.
* `reports/`: Relatórios gerados em fase (incluindo o Relatório de Laboratório Técnico para a Diretoria).

##  Relatório Técnico
O detalhamento algorítmico global, o comportamento frente a *outliers* ruidosos das instituições e os valores quantificáveis dos limites do MAE e RMSE estão formalmente documentados no PDF contido em `reports/Relatorio_Executivo_Diretoria.pdf`.
