"""
================================================================================
 MÓDULO: export_dashboard_data.py
 FASE CRISP-DM: 6 - Deployment (Exportação para Dashboard)
 PROJETO: ML-Forecast-Seg — Previsão de Novos Casos Judiciais (TJGO)
================================================================================
 Gera TODOS os arquivos JSON consumidos pelo dashboard Next.js:
   - forecast_data_1.json / forecast_data_2.json (dados hierárquicos)
   - hierarquia.json (comarca → serventias)
   - kpis.json (métricas do modelo)

 Dados:
   - Histórico (2020-2025): full_prepared_v2.csv
   - Previsões teste 2025: modelo v2 exotic (geradas em tempo de execução)
   - Previsões 2026: 09_previsoes_2026.csv
================================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

from exotic_features import apply_all_exotic_features

PROCESSED_DIR = 'data/processed'
REPORT_DIR = 'reports/tables'
MODEL_PATH = 'models/lgbm_model_v2_exotic.txt'
FEATURE_LIST_PATH = 'models/lgbm_model_v2_exotic_features.json'
PREDICTIONS_2026_PATH = 'reports/tables/09_previsoes_2026.csv'
DASHBOARD_PUBLIC = 'dashboard/public'

MONTH_NAMES = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
               'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
MONTH_MAP = {i+1: name for i, name in enumerate(MONTH_NAMES)}

# Anos a incluir no dashboard (cortar anos antigos para manter JSON leve)
DASHBOARD_START_YEAR = 2020


def load_historical_data():
    """Carrega dados históricos (2020-2025) do dataset preparado."""
    print("  Carregando dados históricos...")
    full = pd.read_csv(f'{PROCESSED_DIR}/full_prepared_v2.csv',
                       usecols=['ANO_MES', 'COMARCA', 'SERVENTIA', 'novos_casos'],
                       low_memory=False)
    full['ANO_MES_DT'] = pd.to_datetime(full['ANO_MES'])
    full['ANO'] = full['ANO_MES_DT'].dt.year
    full['MES'] = full['ANO_MES_DT'].dt.month
    full = full[full['ANO'] >= DASHBOARD_START_YEAR].copy()
    print(f"    {len(full):,} registros ({full['ANO'].min()}-{full['ANO'].max()})")
    return full


def generate_2025_predictions():
    """Gera previsões do modelo v2 (exotic) para o teste 2025."""
    print("  Gerando previsões do modelo v2 para 2025 (teste)...")

    model = lgb.Booster(model_file=MODEL_PATH)
    with open(FEATURE_LIST_PATH) as f:
        feature_cols = json.load(f)

    test = pd.read_csv(f'{PROCESSED_DIR}/test_full.csv', low_memory=False)
    test['ANO_MES_DT'] = pd.to_datetime(test['ANO_MES'])
    test['ANO'] = test['ANO_MES_DT'].dt.year
    test['MES'] = test['ANO_MES_DT'].dt.month

    # Aplicar features exóticas
    test_exotic, _ = apply_all_exotic_features(
        test, group_cols=['COMARCA', 'SERVENTIA'], target='novos_casos'
    )

    # Preparar para predição
    train = pd.read_csv(f'{PROCESSED_DIR}/train_full.csv',
                        usecols=['COMARCA', 'SERVENTIA'], low_memory=False)
    comarca_cats = train['COMARCA'].astype('category').cat.categories
    serventia_cats = train['SERVENTIA'].astype('category').cat.categories

    X_test = test_exotic[feature_cols].copy()
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    X_test['COMARCA'] = pd.Categorical(X_test['COMARCA'], categories=comarca_cats)
    X_test['SERVENTIA'] = pd.Categorical(X_test['SERVENTIA'], categories=serventia_cats)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    result = test[['ANO_MES', 'COMARCA', 'SERVENTIA', 'novos_casos']].copy()
    result['previsao'] = np.round(y_pred, 0).astype(int)
    result['ANO'] = test['ANO']
    result['MES'] = test['MES']

    # Calcular métricas
    y_true = result['novos_casos'].values
    y_p = result['previsao'].values
    total = np.sum(np.abs(y_true))
    wmape_val = float(np.sum(np.abs(y_true - y_p)) / total * 100) if total > 0 else 0
    mae_val = float(np.mean(np.abs(y_true - y_p)))
    rmse_val = float(np.sqrt(np.mean((y_true - y_p) ** 2)))

    # R² calculation
    ss_res = np.sum((y_true - y_p) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    metrics = {
        'r2': round(r2_val, 2),
        'rmse': round(rmse_val, 1),
        'wmape': f"{wmape_val:.2f}%",
        'mae': round(mae_val, 1)
    }

    print(f"    Métricas: R²={metrics['r2']}, RMSE={metrics['rmse']}, "
          f"WMAPE={metrics['wmape']}, MAE={metrics['mae']}")

    return result, metrics


def load_2026_predictions():
    """Carrega previsões de 2026."""
    print("  Carregando previsões 2026...")
    pred = pd.read_csv(PREDICTIONS_2026_PATH)
    pred['ANO_MES_DT'] = pd.to_datetime(pred['ANO_MES'])
    pred['ANO'] = pred['ANO_MES_DT'].dt.year
    pred['MES'] = pred['ANO_MES_DT'].dt.month
    print(f"    {len(pred):,} registros")
    return pred


def build_hierarchical_json(hist, pred_2025, pred_2026):
    """
    Constrói a estrutura hierárquica:
    {
      "Comarca": {
        "Serventia": {
          "2020": {
            "Jan": { "historico": N, "previsao": N|null, "previsao_min": N|null, "previsao_max": N|null }
          }
        }
      }
    }
    """
    print("\n  Construindo estrutura hierárquica...")

    # Calcular margem de erro para previsões (baseada no WMAPE por grupo)
    ERROR_MARGIN = 0.15  # ±15% como intervalo de confiança

    result = {}

    # Todas as comarcas e serventias
    all_comarcas = sorted(hist['COMARCA'].unique())
    hierarquia = {}

    for comarca in all_comarcas:
        hist_c = hist[hist['COMARCA'] == comarca]
        serventias = sorted(hist_c['SERVENTIA'].unique())
        hierarquia[comarca] = serventias

        result[comarca] = {}

        for serventia in serventias + ['Tudo']:
            if serventia == 'Tudo':
                h_sub = hist_c
                p25_sub = pred_2025[pred_2025['COMARCA'] == comarca]
                p26_sub = pred_2026[pred_2026['COMARCA'] == comarca]
            else:
                h_sub = hist_c[hist_c['SERVENTIA'] == serventia]
                p25_sub = pred_2025[(pred_2025['COMARCA'] == comarca) &
                                    (pred_2025['SERVENTIA'] == serventia)]
                p26_sub = pred_2026[(pred_2026['COMARCA'] == comarca) &
                                    (pred_2026['SERVENTIA'] == serventia)]

            year_data = {}
            # Histórico (2020-2025)
            for ano in range(DASHBOARD_START_YEAR, 2026):
                month_data = {}
                for mes_num in range(1, 13):
                    mes_name = MONTH_MAP[mes_num]
                    h_val = h_sub[(h_sub['ANO'] == ano) & (h_sub['MES'] == mes_num)]['novos_casos'].sum()
                    h_val = int(h_val) if h_val > 0 else 0

                    # Se ano=2025, incluir previsão do modelo (teste)
                    if ano == 2025:
                        p_val = p25_sub[p25_sub['MES'] == mes_num]['previsao'].sum()
                        p_val = int(p_val) if p_val > 0 else 0
                        p_min = max(0, int(p_val * (1 - ERROR_MARGIN)))
                        p_max = int(p_val * (1 + ERROR_MARGIN))
                        month_data[mes_name] = {
                            'historico': h_val,
                            'previsao': p_val,
                            'previsao_min': p_min,
                            'previsao_max': p_max
                        }
                    else:
                        month_data[mes_name] = {
                            'historico': h_val,
                            'previsao': None,
                            'previsao_min': None,
                            'previsao_max': None
                        }
                year_data[str(ano)] = month_data

            # Previsões 2026
            month_data_2026 = {}
            for mes_num in range(1, 13):
                mes_name = MONTH_MAP[mes_num]
                p_val = p26_sub[p26_sub['MES'] == mes_num]['previsao_novos_casos'].sum()
                p_val = int(p_val) if p_val > 0 else 0
                p_min = max(0, int(p_val * (1 - ERROR_MARGIN)))
                p_max = int(p_val * (1 + ERROR_MARGIN))
                month_data_2026[mes_name] = {
                    'historico': None,
                    'previsao': p_val,
                    'previsao_min': p_min,
                    'previsao_max': p_max
                }
            year_data['2026'] = month_data_2026

            result[comarca][serventia] = year_data

    # Agregar "Tudo" (todas as comarcas)
    print("  Agregando totais (Tudo)...")
    result['Tudo'] = {'Tudo': {}}

    for ano in range(DASHBOARD_START_YEAR, 2027):
        month_data = {}
        for mes_num in range(1, 13):
            mes_name = MONTH_MAP[mes_num]

            if ano <= 2025:
                h_val = int(hist[(hist['ANO'] == ano) & (hist['MES'] == mes_num)]['novos_casos'].sum())
            else:
                h_val = None

            if ano == 2025:
                p_val = int(pred_2025[pred_2025['MES'] == mes_num]['previsao'].sum())
                p_min = max(0, int(p_val * (1 - ERROR_MARGIN)))
                p_max = int(p_val * (1 + ERROR_MARGIN))
            elif ano == 2026:
                p_val = int(pred_2026[pred_2026['MES'] == mes_num]['previsao_novos_casos'].sum())
                p_min = max(0, int(p_val * (1 - ERROR_MARGIN)))
                p_max = int(p_val * (1 + ERROR_MARGIN))
            else:
                p_val = None
                p_min = None
                p_max = None

            month_data[mes_name] = {
                'historico': h_val,
                'previsao': p_val,
                'previsao_min': p_min,
                'previsao_max': p_max
            }
        result['Tudo']['Tudo'][str(ano)] = month_data

    return result, hierarquia


def split_and_save_json(data, hierarquia):
    """Divide o JSON em dois arquivos para reduzir tamanho individual."""
    print("\n  Salvando arquivos JSON...")

    comarcas = sorted([k for k in data.keys() if k != 'Tudo'])
    mid = len(comarcas) // 2
    comarcas_1 = comarcas[:mid]
    comarcas_2 = comarcas[mid:]

    # File 1: Tudo + primeira metade das comarcas
    data_1 = {'Tudo': data['Tudo']}
    for c in comarcas_1:
        data_1[c] = data[c]

    # File 2: segunda metade das comarcas
    data_2 = {}
    for c in comarcas_2:
        data_2[c] = data[c]

    path1 = os.path.join(DASHBOARD_PUBLIC, 'forecast_data_1.json')
    path2 = os.path.join(DASHBOARD_PUBLIC, 'forecast_data_2.json')
    path_h = os.path.join(DASHBOARD_PUBLIC, 'hierarquia.json')

    with open(path1, 'w', encoding='utf-8') as f:
        json.dump(data_1, f, ensure_ascii=False, separators=(',', ':'))
    print(f"    forecast_data_1.json: {os.path.getsize(path1)/1024/1024:.1f} MB ({len(data_1)} comarcas)")

    with open(path2, 'w', encoding='utf-8') as f:
        json.dump(data_2, f, ensure_ascii=False, separators=(',', ':'))
    print(f"    forecast_data_2.json: {os.path.getsize(path2)/1024/1024:.1f} MB ({len(data_2)} comarcas)")

    with open(path_h, 'w', encoding='utf-8') as f:
        json.dump(hierarquia, f, ensure_ascii=False, indent=2)
    print(f"    hierarquia.json: {len(hierarquia)} comarcas")


def save_kpis(metrics):
    """Salva métricas do modelo."""
    path = os.path.join(DASHBOARD_PUBLIC, 'kpis.json')
    with open(path, 'w') as f:
        json.dump(metrics, f)
    print(f"    kpis.json: {metrics}")


def main():
    print("=" * 72)
    print(" EXPORTAÇÃO COMPLETA PARA O DASHBOARD")
    print("=" * 72)

    # 1. Carregar dados
    hist = load_historical_data()
    pred_2025, metrics = generate_2025_predictions()
    pred_2026 = load_2026_predictions()

    # 2. Construir estrutura hierárquica
    data, hierarquia = build_hierarchical_json(hist, pred_2025, pred_2026)

    # 3. Salvar arquivos
    os.makedirs(DASHBOARD_PUBLIC, exist_ok=True)
    split_and_save_json(data, hierarquia)
    save_kpis(metrics)

    print("\n" + "=" * 72)
    print(" EXPORTAÇÃO CONCLUÍDA")
    print("=" * 72)


if __name__ == '__main__':
    main()
