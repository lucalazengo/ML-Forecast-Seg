"""
================================================================================
 MÓDULO: train_lgbm.py
 FASE CRISP-DM: 4/6 - Modeling & Deployment (Evolução do Modelo)
 PROJETO: ML-Forecast-Seg — Previsão de Novos Casos Judiciais (TJGO)
================================================================================
 Treina o modelo LightGBM para previsão de novos casos judiciais.
 Diferente da regressão linear OLS, o LightGBM consegue capturar não-linearidades
 e interações complexas. Adicionalmente, utilizamos COMARCA e SERVENTIA como
 features categóricas nativas, permitindo aprendizado de padrões locais.
================================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb

PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models'
REPORT_DIR = 'reports'
IMG_DIR = 'reports/images'
TBL_DIR = 'reports/tables'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# Features Numéricas (as mesmas do modelo Linear)
NUM_FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
    'rolling_std_3',
    'mes_do_ano', 'trimestre', 'is_recesso', 'is_pandemia',
    'mes_sin', 'mes_cos', 'area_civel'
]

# Novas Features Categóricas para o LightGBM
CAT_COLS = ['COMARCA', 'SERVENTIA']

FEATURE_COLS = NUM_FEATURE_COLS + CAT_COLS
TARGET = 'novos_casos'


# ─── MÉTRICAS ────────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def wmape(y_true, y_pred):
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float('nan')
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


# ─── CARREGAMENTO DOS DADOS ───────────────────────────────────────────────────

def load_data():
    print("\n" + "=" * 72)
    print(" [1/4] CARREGANDO DATASETS PROCESSADOS (Para LightGBM)")
    print("=" * 72)

    train = pd.read_csv(f'{PROCESSED_DIR}/train_full.csv', low_memory=False)
    test  = pd.read_csv(f'{PROCESSED_DIR}/test_full.csv',  low_memory=False)

    # Garantir numéricos
    for col in NUM_FEATURE_COLS + [TARGET]:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        test[col]  = pd.to_numeric(test[col],  errors='coerce')

    train = train.dropna(subset=NUM_FEATURE_COLS + [TARGET])
    test  = test.dropna(subset=NUM_FEATURE_COLS + [TARGET])

    # Configurar Categóricas (O LightGBM precisa que sejam do tipo 'category' do Pandas)
    for col in CAT_COLS:
        train[col] = train[col].astype('category')
        # Garantir que o teste tenha as mesmas categorias do treino
        test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

    print(f"  Treino: {len(train):,} linhas | {train['ANO_MES'].min()} → {train['ANO_MES'].max()}")
    print(f"  Teste:  {len(test):,} linhas  | {test['ANO_MES'].min()} → {test['ANO_MES'].max()}")
    print(f"  Features: {len(FEATURE_COLS)} (sendo {len(CAT_COLS)} categóricas)")
    
    return train, test


# ─── TREINAMENTO DO MODELO LIGHTGBM ───────────────────────────────────────────

def train_lgbm(train, test):
    print("\n" + "=" * 72)
    print(" [2/4] TREINANDO MODELO M3 — LIGHTGBM REGRESSOR")
    print("=" * 72)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test  = test[FEATURE_COLS]
    y_true  = test[TARGET].values

    print(f"  Iniciando treinamento com {len(X_train):,} amostras...")
    
    # Hiperparâmetros recomendados para contagem robusta e evitar overfitting
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Treinamento
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_true)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)]
    )

    print("\n  Treinamento concluído. Gerando previsões...")
    y_pred = model.predict(X_test)
    
    # Prevenir previsões negativas (comum em regressão para séries de contagem)
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        'modelo': 'M3 — LightGBM',
        'MAE':   round(mae(y_true, y_pred), 2),
        'RMSE':  round(rmse(y_true, y_pred), 2),
        'WMAPE': round(wmape(y_true, y_pred), 2),
    }
    
    print("\n  Métricas do LightGBM no Teste (2024):")
    print(f"  MAE:   {metrics['MAE']:>10.2f}")
    print(f"  RMSE:  {metrics['RMSE']:>10.2f}")
    print(f"  WMAPE: {metrics['WMAPE']:>9.2f}%")

    return model, y_pred, metrics


# ─── VISUALIZAÇÕES E EXPORTAÇÕES ──────────────────────────────────────────────

def plot_lgbm_feature_importance(model, feature_names):
    """Gráfico de importância das variáveis no LightGBM."""
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importância': importances
    }).sort_values('Importância', ascending=True).tail(15) # Top 15

    fig = px.bar(
        imp_df, y='Feature', x='Importância',
        orientation='h',
        title='<b>Top 15 Features — M3 LightGBM</b>',
        color='Importância', color_continuous_scale='Blues',
    )
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=12),
        height=550, showlegend=False,
    )
    path = f'{IMG_DIR}/14_lgbm_feature_importance.html'
    fig.write_html(path)
    print(f"  ✔ Gráfico feature importance salvo: {path}")


def export_predictions(test_df, y_pred):
    """Atualiza a tabela de previsões com o resultado do LightGBM."""
    path = f'{TBL_DIR}/07_previsoes_2024.csv'
    
    # Se a tabela da fase 4 existir, fazemos o merge, senão criamos uma nova
    if os.path.exists(path):
        out = pd.read_csv(path)
        out['previsto_m3_lgbm'] = np.round(y_pred, 1)
        # Recalcular erro baseando-se no novo melhor modelo se ele for melhor
        out['erro_lgbm'] = np.round(out[TARGET] - out['previsto_m3_lgbm'], 1)
    else:
        out = test_df[['ANO_MES', 'COMARCA', 'SERVENTIA', TARGET]].copy()
        out['previsto_m3_lgbm'] = np.round(y_pred, 1)
        out['erro_lgbm'] = np.round(out[TARGET] - out['previsto_m3_lgbm'], 1)
        
    out.to_csv(path, index=False)
    print(f"  ✔ Tabela de previsões atualizada: {path}")


def export_metrics(metrics):
    """Adiciona as métricas do LightGBM ao arquivo de comparação."""
    path = f'{TBL_DIR}/08_metricas_modelos.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Remove se já existir para não duplicar
        df = df[df['modelo'] != metrics['modelo']]
        df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics])
        
    df.to_csv(path, index=False)
    print(f"  ✔ Tabela de métricas atualizada: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print(" CRISP-DM | Fase 4/6 — Evolução do Modelo (LightGBM)")
    print(" Adicionando COMARCA e SERVENTIA como features categóricas.")
    print("=" * 72)

    # 1. Carregar dados com encoding categórico
    train, test = load_data()

    # 2. Treinar Modelo
    model, y_pred, metrics = train_lgbm(train, test)

    # 3. Exportar resultados e gerar gráficos
    print("\n" + "=" * 72)
    print(" [3/4] EXPORTANDO RESULTADOS")
    print("=" * 72)
    
    plot_lgbm_feature_importance(model, FEATURE_COLS)
    export_predictions(test, y_pred)
    export_metrics(metrics)
    
    # Salvar o modelo fisicamente
    model.booster_.save_model(f'{MODEL_DIR}/lgbm_model_v1.txt')
    print(f"  ✔ Modelo salvo em: {MODEL_DIR}/lgbm_model_v1.txt")

    print("\n" + "=" * 72)
    print(" ✅ TREINAMENTO LIGHTGBM CONCLUÍDO")
    print(f"    WMAPE do LightGBM alcançado: {metrics['WMAPE']:.2f}%")
    print("    Execute o src/generate_dashboard.py para atualizar os gráficos gerais!")
    print("=" * 72)


if __name__ == '__main__':
    main()