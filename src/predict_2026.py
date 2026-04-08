"""
================================================================================
 MÓDULO: predict_2026.py
 FASE CRISP-DM: 6 - Deployment (Inferência)
 PROJETO: ML-Forecast-Seg — Previsão de Novos Casos Judiciais (TJGO)
================================================================================
 Executa a Previsão Recursiva (Recursive Forecasting) para 2026.
 O algoritmo prevê um mês, retroalimenta a base de dados com a previsão,
 recalcula os Lags e Rolling Means, e prevê o mês subsequente.
================================================================================
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Importar módulo de features exóticas
from exotic_features import apply_all_exotic_features

PROCESSED_DIR = 'data/processed'
MODEL_PATH = 'models/lgbm_model_v2_exotic.txt'  # Novo modelo com features exóticas
FEATURE_LIST_PATH = 'models/lgbm_model_v2_exotic_features.json'
OUTPUT_PATH = 'reports/tables/09_previsoes_2026.csv'

# Carregar feature list do modelo treinado
with open(FEATURE_LIST_PATH, 'r') as f:
    FEATURE_COLS = json.load(f)

def get_original_categories():
    """Garante que as categorias usadas no Teste/Treino sejam as mesmas na Inferência."""
    train = pd.read_csv(f'{PROCESSED_DIR}/train_full.csv', usecols=['COMARCA', 'SERVENTIA'], low_memory=False)
    comarca_cats = train['COMARCA'].astype('category').cat.categories
    serventia_cats = train['SERVENTIA'].astype('category').cat.categories
    return comarca_cats, serventia_cats

def generate_calendar_features(df):
    df['ANO_MES_DT'] = pd.to_datetime(df['ANO_MES'])
    df['ANO'] = df['ANO_MES_DT'].dt.year
    df['MES'] = df['ANO_MES_DT'].dt.month
    
    df['mes_do_ano'] = df['MES']
    df['trimestre'] = ((df['MES'] - 1) // 3) + 1
    df['is_recesso'] = df['MES'].isin([1, 7]).astype(int)
    df['is_pandemia'] = 0  # Em 2026 não consideramos pandemia
    
    df['mes_sin'] = np.sin(2 * np.pi * df['MES'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['MES'] / 12)
    df['area_civel'] = (df['area_predominante'] == 'Cível').astype(int)
    return df

def predict_future_recursive(horizon=12):
    print("=" * 72)
    print(f" INICIANDO INFERÊNCIA RECURSIVA — {horizon} MESES (2026)")
    print("=" * 72)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}.")
        print("Você executou o script src/train_lgbm.py?")
        return

    print("  📂 Carregando histórico recente e modelo LightGBM...")
    # Carregamos o dataset preparado, limitando a partir de 2024 para economizar memória 
    # (Precisamos de no máximo 12 meses para o lag_12)
    hist = pd.read_csv(f'{PROCESSED_DIR}/full_prepared_v2.csv', low_memory=False)
    hist['ANO_MES_DT'] = pd.to_datetime(hist['ANO_MES'])
    hist = hist[hist['ANO_MES_DT'].dt.year >= 2024].copy()
    hist = hist[['ANO_MES', 'COMARCA', 'SERVENTIA', 'novos_casos', 'area_predominante']]
    
    model = lgb.Booster(model_file=MODEL_PATH)
    comarca_cats, serventia_cats = get_original_categories()

    # Extrai as combinações únicas (Comarcas x Serventias x Área) do nosso acervo
    pairs = hist[['COMARCA', 'SERVENTIA', 'area_predominante']].drop_duplicates()
    
    # Cria a lista de meses a prever
    future_months = [f'2026-{str(m).zfill(2)}' for m in range(1, horizon + 1)]
    
    df = hist.copy()
    
    print(f"\n  🔄 Previsão Mês a Mês (Retroalimentando resultados com Exotic Features):")
    for month in future_months:
        print(f"     ➔ Projetando: {month}...")

        # Prepara o grid de serventias para o mês alvo
        current_month_df = pairs.copy()
        current_month_df['ANO_MES'] = month
        current_month_df['novos_casos'] = np.nan # Ainda não sabemos

        df = pd.concat([df, current_month_df], ignore_index=True)
        df = df.sort_values(['COMARCA', 'SERVENTIA', 'ANO_MES']).reset_index(drop=True)

        # Otimização: Manter apenas os últimos 24 meses em cálculo para não sobrecarregar
        if len(df) > (24 * len(pairs)):
            df = df.groupby(['COMARCA', 'SERVENTIA']).tail(24).reset_index(drop=True)

        # Recalcular as features usando o histórico + previsões que já fizemos
        g = df.groupby(['COMARCA', 'SERVENTIA'])['novos_casos']

        for lag in [1, 2, 3, 6, 12]:
            df[f'lag_{lag}'] = g.shift(lag)

        for w in [3, 6, 12]:
            df[f'rolling_mean_{w}'] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

        df['rolling_std_3'] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())

        df = generate_calendar_features(df)

        # ✨ Aplicar features exóticas
        df_with_exotic, _ = apply_all_exotic_features(
            df,
            group_cols=['COMARCA', 'SERVENTIA'],
            target='novos_casos'
        )

        # Filtrar o mês que estamos tentando prever
        mask = df_with_exotic['ANO_MES'] == month

        # Selecionar APENAS as features que o modelo espera (na ordem correta)
        X_pred = df_with_exotic.loc[mask, FEATURE_COLS].copy()

        # Preencher NaNs com 0
        X_pred = X_pred.fillna(0).replace([np.inf, -np.inf], 0)

        # Aplicar Encoding Categórico idêntico ao Treino
        X_pred['COMARCA'] = pd.Categorical(X_pred['COMARCA'], categories=comarca_cats)
        X_pred['SERVENTIA'] = pd.Categorical(X_pred['SERVENTIA'], categories=serventia_cats)

        # Fazer Previsão
        try:
            preds = model.predict(X_pred)
            preds = np.clip(preds, 0, None) # Não existem casos negativos

            # Atualizar a tabela principal para que o próximo mês possa usar essa previsão como 'lag'
            df.loc[mask, 'novos_casos'] = preds
        except Exception as e:
            print(f"     ❌ Erro na predição: {str(e)[:100]}")
            print(f"     Expected features: {len(FEATURE_COLS)}, Got: {X_pred.shape[1]}")
            raise
        
    print("\n  💾 Exportando resultados...")
    # Filtra apenas o ano de 2026 para salvar
    preds_2026 = df[df['ANO_MES'].str.startswith('2026')].copy()
    preds_2026['novos_casos'] = np.round(preds_2026['novos_casos'], 0).astype(int)
    
    out = preds_2026[['ANO_MES', 'COMARCA', 'SERVENTIA', 'novos_casos']].rename(columns={'novos_casos': 'previsao_novos_casos'})
    out.to_csv(OUTPUT_PATH, index=False)
    
    # Resumo executivo rápido no Terminal
    resumo_estado = out.groupby('ANO_MES')['previsao_novos_casos'].sum().reset_index()
    print("\n" + "=" * 72)
    print(" 📊 RESUMO DE PREVISÕES AGREGADAS (ESTADO DE GOIÁS — 2026)")
    print("-" * 72)
    for _, row in resumo_estado.iterrows():
        print(f"     Mês: {row['ANO_MES']} | Previsão Total: {row['previsao_novos_casos']:,} novos casos")
    print("-" * 72)
    print(f"     Total 1º Semestre (6 Meses): {resumo_estado.head(6)['previsao_novos_casos'].sum():,}")
    print(f"     Total Anual (12 Meses):      {resumo_estado['previsao_novos_casos'].sum():,}")
    print("=" * 72)
    
    print(f"\n✅ Tabela granular completa salva em: {OUTPUT_PATH}")

if __name__ == '__main__':
    predict_future_recursive(horizon=12)
