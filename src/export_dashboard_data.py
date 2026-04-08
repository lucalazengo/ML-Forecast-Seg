"""
================================================================================
 MÓDULO: export_dashboard_data.py
 FASE CRISP-DM: 6 - Deployment (Exportação para Dashboard)
 PROJETO: ML-Forecast-Seg — Previsão de Novos Casos Judiciais (TJGO)
================================================================================
 Carrega as previsões geradas por `predict_2026.py` e as exporta para o
 diretório público do dashboard Next.js em formato JSON.
================================================================================
"""
import os
import pandas as pd
import json

# Define paths
PREDICTIONS_CSV_PATH = 'reports/tables/09_previsoes_2026.csv'
DASHBOARD_DATA_DIR = 'dashboard/public/data'
DASHBOARD_JSON_PATH = os.path.join(DASHBOARD_DATA_DIR, 'forecast_data.json')

def export_predictions_to_dashboard():
    """
    Carrega as previsões de 2026, formata-as e as exporta para
    o diretório público do dashboard como um arquivo JSON.
    """
    print("=" * 72)
    print(" EXPORTANDO DADOS DE PREVISÃO PARA O DASHBOARD")
    print("=" * 72)

    if not os.path.exists(PREDICTIONS_CSV_PATH):
        print(f"❌ ERRO: Arquivo de previsões não encontrado em {PREDICTIONS_CSV_PATH}.")
        print("Certifique-se de que 'src/predict_2026.py' foi executado com sucesso.")
        return

    print(f"  📂 Carregando previsões de: {PREDICTIONS_CSV_PATH}")
    df_predictions = pd.read_csv(PREDICTIONS_CSV_PATH)

    # Converter para o formato JSON esperado pelo dashboard (lista de objetos)
    dashboard_data = df_predictions.to_dict(orient='records')

    # Criar o diretório de destino se não existir
    os.makedirs(DASHBOARD_DATA_DIR, exist_ok=True)

    print(f"  💾 Salvando dados formatados em: {DASHBOARD_JSON_PATH}")
    with open(DASHBOARD_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Exportação para o dashboard concluída com sucesso!")
    print(f"   O dashboard agora deve consumir os dados de: {DASHBOARD_JSON_PATH}")
    print("=" * 72)

if __name__ == '__main__':
    export_predictions_to_dashboard()