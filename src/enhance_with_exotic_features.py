"""
================================================================================
 SCRIPT: enhance_with_exotic_features.py
 APLICAÇÃO: Enriquecimento de Features e Retreinamento do Modelo
================================================================================
 Pipeline:
 1. Carregar dados preparados
 2. Aplicar todas as técnicas de feature engineering exótica
 3. Reexportar datasets enriquecidos
 4. Retreinar LightGBM com as novas features
 5. Comparar WMAPE com baseline (M3)
================================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from exotic_features import apply_all_exotic_features

PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models'
REPORT_DIR = 'reports/tables'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def wmape(y_true, y_pred):
    """Weighted Mean Absolute Percentage Error."""
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float('nan')
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def step_1_load_prepared_data():
    """Carrega os dados já preparados (phase 3)."""
    print("\n" + "=" * 72)
    print(" [1/4] CARREGANDO DADOS PREPARADOS")
    print("=" * 72)

    train = pd.read_csv(f'{PROCESSED_DIR}/train_full.csv', low_memory=False)
    test = pd.read_csv(f'{PROCESSED_DIR}/test_full.csv', low_memory=False)
    full = pd.read_csv(f'{PROCESSED_DIR}/full_prepared_v2.csv', low_memory=False)

    # Converter ANO_MES_DT para datetime
    for df in [train, test, full]:
        df['ANO_MES_DT'] = pd.to_datetime(df['ANO_MES'])
        df['ANO'] = df['ANO_MES_DT'].dt.year
        df['MES'] = df['ANO_MES_DT'].dt.month

    print(f"  ✔ Treino: {len(train):,} linhas")
    print(f"  ✔ Teste:  {len(test):,} linhas")
    print(f"  ✔ Completo: {len(full):,} linhas")

    return train, test, full


def step_2_apply_exotic_features(df, name=''):
    """Aplica todas as features exóticas ao DataFrame."""
    print(f"\n{name}")
    df_enhanced, new_features = apply_all_exotic_features(
        df,
        group_cols=['COMARCA', 'SERVENTIA'],
        target='novos_casos'
    )
    return df_enhanced, new_features


def step_3_prepare_for_training(train, test, new_features):
    """Prepara os dados para treinamento, removendo NaNs introduzidos pelas novas features."""
    print("\n" + "=" * 72)
    print(" [2/4] PREPARANDO DATASETS PARA TREINAMENTO")
    print("=" * 72)

    # Features originais (que usamos antes)
    original_num_features = [
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3',
        'mes_do_ano', 'trimestre', 'is_recesso', 'is_pandemia',
        'mes_sin', 'mes_cos', 'area_civel'
    ]

    # Features categóricas
    cat_features = ['COMARCA', 'SERVENTIA']

    # Remover duplicatas (new_features pode conter originais)
    clean_new_features = [f for f in new_features if f not in original_num_features]

    # Combinar: original + exótica (únicas)
    all_features = original_num_features + clean_new_features + cat_features

    # Remover features que causam problemas
    problematic = [c for c in all_features if c not in train.columns]
    if problematic:
        print(f"  ⚠️ Removendo features não encontradas: {problematic}")
        all_features = [f for f in all_features if f in train.columns]

    # Remove duplicatas finais
    all_features = list(dict.fromkeys(all_features))  # Remove duplicatas mantendo ordem

    # Remover linhas com NaN em features críticas
    critical_features = original_num_features + ['novos_casos']
    n_before_train = len(train)
    train = train.dropna(subset=critical_features)
    n_after_train = len(train)

    n_before_test = len(test)
    test = test.dropna(subset=critical_features)
    n_after_test = len(test)

    print(f"  → Treino após remover NaNs: {n_before_train:,} → {n_after_train:,} ({n_before_train - n_after_train} removidas)")
    print(f"  → Teste após remover NaNs:  {n_before_test:,} → {n_after_test:,} ({n_before_test - n_after_test} removidas)")

    # Preencher NaNs em features exóticas com 0
    for col in all_features:
        if col in train.columns:
            train[col] = train[col].fillna(0).replace([np.inf, -np.inf], 0)
        if col in test.columns:
            test[col] = test[col].fillna(0).replace([np.inf, -np.inf], 0)

    print(f"  → Total de features para treinamento: {len(all_features)}")
    print(f"     Originais: {len(original_num_features)} | Exóticas: {len(new_features)} | Categóricas: {len(cat_features)}")

    return train, test, all_features, cat_features


def step_4_train_model(train, test, feature_cols, cat_features):
    """Treina o modelo LightGBM com as novas features."""
    print("\n" + "=" * 72)
    print(" [3/4] TREINANDO LIGHTGBM COM EXOTIC FEATURES")
    print("=" * 72)

    X_train = train[feature_cols]
    y_train = train['novos_casos'].values
    X_test = test[feature_cols]
    y_true = test['novos_casos'].values

    print(f"  → Amostras de treino: {len(X_train):,}")
    print(f"  → Amostras de teste:  {len(X_test):,}")
    print(f"  → Features numéricas: {len([f for f in feature_cols if f not in cat_features])}")
    print(f"  → Features categóricas: {len(cat_features)}")

    # Configurar categorias
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=400,  # Aumentado para capturar features complexas
        learning_rate=0.04,  # Ligeiramente mais baixo para convergência estável
        max_depth=10,  # Um pouco mais profundo
        num_leaves=95,
        subsample=0.75,  # Mais regularização
        colsample_bytree=0.75,
        reg_alpha=0.5,  # Regularização L1
        reg_lambda=1.0,  # Regularização L2
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    print(f"\n  Iniciando treinamento...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_true)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        'modelo': 'M4 — LightGBM + Exotic Features',
        'MAE': round(mae(y_true, y_pred), 2),
        'RMSE': round(rmse(y_true, y_pred), 2),
        'WMAPE': round(wmape(y_true, y_pred), 2),
    }

    print(f"\n  ✅ Métricas do Modelo Enriquecido:")
    print(f"     MAE:   {metrics['MAE']:>10.2f}")
    print(f"     RMSE:  {metrics['RMSE']:>10.2f}")
    print(f"     WMAPE: {metrics['WMAPE']:>9.2f}% ⭐")

    return model, y_pred, metrics


def step_5_export_results(test, y_pred, model, feature_cols, metrics):
    """Exporta resultados e comparações."""
    print("\n" + "=" * 72)
    print(" [4/4] EXPORTANDO RESULTADOS")
    print("=" * 72)

    # Salvar modelo
    model.booster_.save_model(f'{MODEL_DIR}/lgbm_model_v2_exotic.txt')
    print(f"  ✔ Modelo salvo: {MODEL_DIR}/lgbm_model_v2_exotic.txt")

    # Salvar lista de features para uso em predict_2026.py
    feature_list = list(feature_cols)
    with open(f'{MODEL_DIR}/lgbm_model_v2_exotic_features.json', 'w') as f:
        json.dump(feature_list, f)
    print(f"  ✔ Feature list salva: {MODEL_DIR}/lgbm_model_v2_exotic_features.json")

    # Feature importance
    importances_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importância': model.feature_importances_
    }).sort_values('Importância', ascending=False)

    import_path = f'{REPORT_DIR}/10_feature_importance_exotic.csv'
    importances_df.to_csv(import_path, index=False)
    print(f"  ✔ Feature importance salvo: {import_path}")

    # Atualizar tabela de métricas
    metrics_path = f'{REPORT_DIR}/08_metricas_modelos.csv'
    try:
        metrics_df = pd.read_csv(metrics_path)
        metrics_df = metrics_df[metrics_df['modelo'] != metrics['modelo']]
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
    except:
        metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(metrics_path, index=False)
    print(f"  ✔ Tabela de métricas atualizada: {metrics_path}")

    # Top 20 features
    print(f"\n  Top 20 Features Mais Importantes:")
    for i, row in importances_df.head(20).iterrows():
        print(f"     {row['Feature']:30s} → {row['Importância']:8.1f}")


def main():
    print("=" * 72)
    print(" PIPELINE: EXOTIC FEATURE ENGINEERING + RETREINAMENTO")
    print("=" * 72)

    # 1. Carregar dados preparados
    train, test, full = step_1_load_prepared_data()

    # 2. Aplicar features exóticas
    train_exotic, new_features_train = step_2_apply_exotic_features(train, '\n  Aplicando features ao Treino...')
    test_exotic, _ = step_2_apply_exotic_features(test, '\n  Aplicando features ao Teste...')

    # 3. Preparar para treinamento
    train_ready, test_ready, all_features, cat_features = step_3_prepare_for_training(
        train_exotic, test_exotic, new_features_train
    )

    # 4. Treinar modelo enriquecido
    model, y_pred, metrics = step_4_train_model(
        train_ready, test_ready, all_features, cat_features
    )

    # 5. Exportar resultados
    step_5_export_results(test_ready, y_pred, model, all_features, metrics)

    print("\n" + "=" * 72)
    print(" ✅ PIPELINE CONCLUÍDO COM SUCESSO")
    print(f"    WMAPE do novo modelo: {metrics['WMAPE']:.2f}%")
    print("=" * 72)


if __name__ == '__main__':
    main()
