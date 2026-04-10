"""
src/models/train_xgboost.py
Treinamento e tuning de XGBoost para previsão de séries temporais.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
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
    'n_estimators'      : 1000,
    'learning_rate'     : 0.05,
    'max_depth'         : 5,
    'min_child_weight'  : 3,
    'subsample'         : 0.8,
    'colsample_bytree'  : 0.8,
    'colsample_bylevel' : 0.8,
    'gamma'             : 0.1,
    'reg_alpha'         : 0.05,    # L1 — sparsidade
    'reg_lambda'        : 1.0,     # L2 — suavidade
    'early_stopping_rounds': 50,
    'eval_metric'       : 'rmse',
    'random_state'      : RANDOM_STATE,
    'n_jobs'            : -1,
    'verbosity'         : 0,
}


def get_tscv():
    """TimeSeriesSplit com gap para evitar leakage."""
    return TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   params: dict = None) -> xgb.XGBRegressor:
    """Treina XGBoost com early stopping no conjunto de validação."""
    p = {**BASE_PARAMS, **(params or {})}

    model = xgb.XGBRegressor(**p)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    logger.info(f"XGBoost treinado. Best iteration: {model.best_iteration}")
    return model


def cross_validate_xgboost(df: pd.DataFrame, feature_cols: list,
                             params: dict = None) -> dict:
    """Validação cruzada temporal walk-forward."""
    tscv = get_tscv()
    X = df[feature_cols]
    y = df[TARGET_COL]

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = train_xgboost(X_tr, y_tr, X_val, y_val, params)
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
    logger.info(f"CV XGBoost — RMSE: {summary['rmse_mean']:.2f} ± {summary['rmse_std']:.2f}")
    return summary


def tune_xgboost_optuna(df: pd.DataFrame, feature_cols: list,
                         n_trials: int = 50) -> dict:
    """Tuning de hiperparâmetros com Optuna e TimeSeriesSplit."""
    X = df[feature_cols]
    y = df[TARGET_COL]
    tscv = get_tscv()

    def objective(trial):
        params = {
            'n_estimators'      : trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate'     : trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'max_depth'         : trial.suggest_int('max_depth', 3, 9),
            'min_child_weight'  : trial.suggest_int('min_child_weight', 1, 10),
            'subsample'         : trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree'  : trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma'             : trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha'         : trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda'        : trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'early_stopping_rounds': 30,
            'eval_metric'       : 'rmse',
            'random_state'      : RANDOM_STATE,
            'verbosity'         : 0,
        }

        fold_rmses = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, y_tr   = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            fold_rmses.append(np.sqrt(np.mean((y_val - preds) ** 2)))

        return np.mean(fold_rmses)

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    logger.info(f"Melhor RMSE (Optuna): {study.best_value:.4f}")
    logger.info(f"Melhores params: {best}")
    return best


def train_final_xgboost(df: pd.DataFrame, feature_cols: list,
                          params: dict = None) -> xgb.XGBRegressor:
    """Treina modelo final em TODO o dataset com melhores params."""
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Sem early stopping no treino final (sem val set)
    p = {**BASE_PARAMS, **(params or {})}
    p.pop('early_stopping_rounds', None)

    model = xgb.XGBRegressor(**p)
    model.fit(X, y, verbose=False)
    return model


def save_model(model: xgb.XGBRegressor, metrics: dict,
               feature_cols: list) -> str:
    """Salva modelo com timestamp e métricas no nome."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts    = datetime.now().strftime('%Y%m%d_%H%M')
    rmse  = round(metrics.get('rmse_mean', 0), 2)
    fname = f"xgb_{ts}_rmse_{rmse}.json"
    fpath = os.path.join(MODELS_DIR, fname)

    model.save_model(fpath)

    # Salvar metadados
    meta = {
        'model_type'   : 'xgboost',
        'trained_at'   : ts,
        'metrics'      : metrics,
        'feature_cols' : feature_cols,
        'horizon'      : HORIZON,
        'n_features'   : len(feature_cols)
    }
    with open(fpath.replace('.json', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Modelo salvo: {fpath}")
    return fpath


def get_feature_importance(model: xgb.XGBRegressor,
                            feature_cols: list) -> pd.Series:
    """Retorna importância das features ordenada."""
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)
    return importance


if __name__ == "__main__":
    from src.features.build_features import get_feature_columns

    df = pd.read_parquet("data/processed/features.parquet")
    feature_cols = get_feature_columns(df)

    # 1. Validação cruzada com params base
    logger.info("=== Cross-validation com params base ===")
    cv_results = cross_validate_xgboost(df, feature_cols)

    # 2. Tuning (opcional — descomente para rodar)
    # logger.info("=== Tuning com Optuna ===")
    # best_params = tune_xgboost_optuna(df, feature_cols, n_trials=50)

    # 3. Treino final
    model = train_final_xgboost(df, feature_cols)
    fpath = save_model(model, cv_results, feature_cols)

    # 4. Feature importance
    importance = get_feature_importance(model, feature_cols)
    print("\nTop 10 features:")
    print(importance.head(10))
