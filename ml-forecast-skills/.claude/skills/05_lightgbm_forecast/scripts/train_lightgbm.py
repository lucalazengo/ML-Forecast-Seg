"""
src/models/train_lightgbm.py
Treinamento e tuning de LightGBM para previsão de séries temporais.
LightGBM difere do XGBoost: cresce leaf-wise (não level-wise),
o que o torna mais rápido e preciso em datasets grandes,
mas exige mais cuidado com overfitting via num_leaves e min_data_in_leaf.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
import joblib, json, logging, sys, os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import TARGET_COL, HORIZON, N_SPLITS, GAP, RANDOM_STATE, MODELS_DIR
from src.utils.metrics import compute_all_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Parâmetros base — ponto de partida seguro
# ──────────────────────────────────────────────
BASE_PARAMS = {
    'n_estimators'        : 1000,
    'learning_rate'       : 0.05,
    'num_leaves'          : 31,        # PRINCIPAL regularizador do LGBM
    'max_depth'           : -1,        # -1 = sem limite (controlado por num_leaves)
    'min_data_in_leaf'    : 20,        # evita folhas com poucos dados
    'feature_fraction'    : 0.8,       # equivalente a colsample_bytree
    'bagging_fraction'    : 0.8,       # equivalente a subsample
    'bagging_freq'        : 5,         # bagging a cada 5 iterações
    'lambda_l1'           : 0.05,      # L1
    'lambda_l2'           : 1.0,       # L2
    'min_gain_to_split'   : 0.01,      # ganho mínimo para criar split
    'random_state'        : RANDOM_STATE,
    'verbose'             : -1,
    'n_jobs'              : -1,
}

CALLBACKS = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=-1),   # silencia output
]


def get_tscv():
    return TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    params: dict = None) -> lgb.LGBMRegressor:
    """Treina LightGBM com early stopping."""
    p = {**BASE_PARAMS, **(params or {})}

    model = lgb.LGBMRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=CALLBACKS
    )
    logger.info(f"LGBM treinado. Best iteration: {model.best_iteration_}")
    return model


def cross_validate_lightgbm(df: pd.DataFrame, feature_cols: list,
                              params: dict = None) -> dict:
    """Validação cruzada temporal walk-forward."""
    tscv = get_tscv()
    X = df[feature_cols]
    y = df[TARGET_COL]

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr   = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx],   y.iloc[val_idx]

        model = train_lightgbm(X_tr, y_tr, X_val, y_val, params)
        preds = model.predict(X_val)
        metrics = compute_all_metrics(y_val, preds)
        metrics['fold'] = fold
        fold_metrics.append(metrics)
        logger.info(f"Fold {fold+1}: RMSE={metrics['rmse']:.2f} MAE={metrics['mae']:.2f} MAPE={metrics['mape']:.2%}")

    results = pd.DataFrame(fold_metrics)
    summary = {
        'rmse_mean' : results['rmse'].mean(),
        'rmse_std'  : results['rmse'].std(),
        'mae_mean'  : results['mae'].mean(),
        'mape_mean' : results['mape'].mean(),
        'n_folds'   : N_SPLITS
    }
    logger.info(f"CV LGBM — RMSE: {summary['rmse_mean']:.2f} ± {summary['rmse_std']:.2f}")
    return summary


def tune_lightgbm_optuna(df: pd.DataFrame, feature_cols: list,
                          n_trials: int = 50) -> dict:
    """
    Tuning de hiperparâmetros LightGBM com Optuna.
    ATENÇÃO: num_leaves e min_data_in_leaf são os mais críticos no LGBM.
    """
    X = df[feature_cols]
    y = df[TARGET_COL]
    tscv = get_tscv()

    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'num_leaves'       : trial.suggest_int('num_leaves', 15, 200),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 5, 100),
            'feature_fraction' : trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq'     : trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1'        : trial.suggest_float('lambda_l1', 1e-4, 10.0, log=True),
            'lambda_l2'        : trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 0.5),
            'random_state'     : RANDOM_STATE,
            'verbose'          : -1,
        }

        fold_rmses = []
        callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
        for train_idx, val_idx in tscv.split(X):
            X_tr, y_tr   = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx],   y.iloc[val_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            preds = model.predict(X_val)
            fold_rmses.append(np.sqrt(np.mean((y_val - preds) ** 2)))

        return np.mean(fold_rmses)

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info(f"Melhor RMSE (Optuna): {study.best_value:.4f}")
    logger.info(f"Melhores params: {best}")
    return best


def train_final_lightgbm(df: pd.DataFrame, feature_cols: list,
                           params: dict = None) -> lgb.LGBMRegressor:
    """Treina modelo final em TODO o dataset."""
    X = df[feature_cols]
    y = df[TARGET_COL]
    p = {**BASE_PARAMS, **(params or {})}

    model = lgb.LGBMRegressor(**p)
    model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
    return model


def save_model(model: lgb.LGBMRegressor, metrics: dict,
               feature_cols: list) -> str:
    """Salva modelo com timestamp e métricas."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts    = datetime.now().strftime('%Y%m%d_%H%M')
    rmse  = round(metrics.get('rmse_mean', 0), 2)
    fname = f"lgbm_{ts}_rmse_{rmse}.pkl"
    fpath = os.path.join(MODELS_DIR, fname)

    joblib.dump(model, fpath)

    meta = {
        'model_type'   : 'lightgbm',
        'trained_at'   : ts,
        'metrics'      : metrics,
        'feature_cols' : feature_cols,
        'horizon'      : HORIZON,
        'n_features'   : len(feature_cols),
        'best_iteration': getattr(model, 'best_iteration_', None)
    }
    with open(fpath.replace('.pkl', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Modelo salvo: {fpath}")
    return fpath


def get_feature_importance(model: lgb.LGBMRegressor,
                            feature_cols: list,
                            importance_type: str = 'gain') -> pd.Series:
    """
    Importância das features.
    importance_type: 'gain' (preferível) ou 'split'
    'gain' = contribuição real para redução do erro
    'split' = frequência de uso (pode inflar features com muitos valores únicos)
    """
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
    logger.info(f"Importância calculada por: {importance_type}")
    return importance


if __name__ == "__main__":
    from src.features.build_features import get_feature_columns

    df = pd.read_parquet("data/processed/features.parquet")
    feature_cols = get_feature_columns(df)

    logger.info("=== Cross-validation com params base ===")
    cv_results = cross_validate_lightgbm(df, feature_cols)

    # logger.info("=== Tuning com Optuna ===")
    # best_params = tune_lightgbm_optuna(df, feature_cols, n_trials=50)

    model = train_final_lightgbm(df, feature_cols)
    fpath = save_model(model, cv_results, feature_cols)

    importance = get_feature_importance(model, feature_cols, importance_type='gain')
    print("\nTop 10 features (gain):")
    print(importance.head(10))
