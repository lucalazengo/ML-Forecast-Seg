# ML-Forecast-Seg — Instruções para Claude Code

## SEMPRE leia primeiro

Antes de qualquer tarefa neste projeto, leia:
`.claude/skills/00_project_context/SKILL.md`

Ela define stack, regras invioláveis e aponta para a skill correta
de acordo com a tarefa solicitada.

## Mapa rápido de skills

| Tarefa                          | Skill                          |
|---------------------------------|--------------------------------|
| Limpar/preparar dados           | `01_data_preprocessing`        |
| Criar features temporais        | `02_feature_engineering`       |
| Detectar outliers               | `03_outlier_anomaly`           |
| Treinar XGBoost                 | `04_xgboost_forecast`          |
| Treinar LightGBM                | `05_lightgbm_forecast`         |
| Avaliar modelo / métricas       | `06_evaluation_metrics`        |
| Organizar pipeline / scripts    | `07_pipeline_structure`        |

## Regras invioláveis

1. Nunca usar dados futuros no treino (lag mínimo = horizon)
2. Validação sempre com `TimeSeriesSplit(gap=HORIZON)`
3. `random_state=42` em todo lugar
4. Comparar sempre com baseline naive antes de reportar resultado
5. Salvar modelos com timestamp e métricas no nome

## Executar pipeline completo

```bash
python run_pipeline.py --model both
```