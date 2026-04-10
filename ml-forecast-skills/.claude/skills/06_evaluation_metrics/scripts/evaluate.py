"""
src/utils/metrics.py + src/models/evaluate.py
Avaliação completa de modelos de forecast com walk-forward validation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import logging, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import TARGET_COL, HORIZON, N_SPLITS, GAP

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray,
         epsilon: float = 1e-8) -> float:
    """MAPE com proteção contra divisão por zero."""
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE — alternativa ao MAPE quando há zeros ou valores próximos a zero."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8
    return float(np.mean(np.abs(y_true - y_pred) / denominator))

def mase(y_true: np.ndarray, y_pred: np.ndarray,
         y_train: np.ndarray, seasonality: int = 1) -> float:
    """
    MASE — Mean Absolute Scaled Error.
    Escala pelo erro do modelo naive sazonal no treino.
    Valor < 1: melhor que naive. Valor > 1: pior que naive.
    """
    naive_errors = np.abs(np.diff(y_train, n=seasonality))
    naive_mae = np.mean(naive_errors) + 1e-8
    return mae(y_true, y_pred) / naive_mae

def compute_all_metrics(y_true, y_pred, y_train=None) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        'rmse' : rmse(y_true, y_pred),
        'mae'  : mae(y_true, y_pred),
        'mape' : mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'n'    : len(y_true),
    }
    if y_train is not None:
        metrics['mase'] = mase(y_true, y_pred, np.array(y_train))

    return metrics


# ──────────────────────────────────────────────
# Baseline naive
# ──────────────────────────────────────────────

def naive_forecast(series: pd.Series, horizon: int = None) -> pd.Series:
    """Naive: repete o último valor observado para todo o horizonte."""
    horizon = horizon or HORIZON
    return series.shift(horizon)

def seasonal_naive_forecast(series: pd.Series,
                             seasonality: int = 7,
                             horizon: int = None) -> pd.Series:
    """Seasonal naive: repete o valor de exatamente 1 período sazonal atrás."""
    horizon = horizon or HORIZON
    return series.shift(seasonality)

def evaluate_baseline(df: pd.DataFrame) -> dict:
    """Avalia naive e seasonal naive para referência."""
    series = df[TARGET_COL]
    naive  = naive_forecast(series)
    s_naive = seasonal_naive_forecast(series)

    valid = ~(naive.isna() | series.isna())
    results = {
        'naive'          : compute_all_metrics(series[valid], naive[valid]),
        'seasonal_naive' : compute_all_metrics(series[valid], s_naive[valid])
    }
    logger.info(f"Baseline RMSE — Naive: {results['naive']['rmse']:.2f} | "
                f"Seasonal Naive: {results['seasonal_naive']['rmse']:.2f}")
    return results


# ──────────────────────────────────────────────
# Walk-forward validation
# ──────────────────────────────────────────────

def walk_forward_evaluate(df: pd.DataFrame, model,
                           feature_cols: list,
                           n_splits: int = None,
                           gap: int = None) -> pd.DataFrame:
    """
    Walk-forward validation: treina no passado, testa no futuro.
    Simula o ambiente real de previsão.
    """
    n_splits = n_splits or N_SPLITS
    gap      = gap      or GAP

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    X = df[feature_cols]
    y = df[TARGET_COL]

    all_results = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr   = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te   = X.iloc[test_idx],  y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        metrics = compute_all_metrics(y_te.values, preds, y_train=y_tr.values)
        metrics['fold']       = fold + 1
        metrics['train_size'] = len(train_idx)
        metrics['test_size']  = len(test_idx)
        metrics['test_start'] = df.index[test_idx[0]]
        metrics['test_end']   = df.index[test_idx[-1]]

        all_results.append(metrics)
        logger.info(f"Fold {fold+1}: RMSE={metrics['rmse']:.2f} "
                    f"MAPE={metrics['mape']:.2%} MASE={metrics.get('mase', '—')}")

    return pd.DataFrame(all_results)


# ──────────────────────────────────────────────
# Análise de resíduos
# ──────────────────────────────────────────────

def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                       lags: int = 14) -> dict:
    """
    Análise completa de resíduos.
    Bons resíduos = ruído branco (sem autocorrelação, média ≈ 0).
    """
    residuals = y_true - y_pred

    # Teste de Ljung-Box — H0: sem autocorrelação
    lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    has_autocorr = (lb_result['lb_pvalue'] < 0.05).any()

    analysis = {
        'mean'          : float(np.mean(residuals)),
        'std'           : float(np.std(residuals)),
        'max_error'     : float(np.max(np.abs(residuals))),
        'ljungbox_pval' : lb_result['lb_pvalue'].min(),
        'has_autocorr'  : has_autocorr,
        'bias'          : float(np.mean(residuals))  # positivo = subestima, negativo = superestima
    }

    if has_autocorr:
        logger.warning("⚠️  Resíduos com autocorrelação — modelo não captura toda a estrutura temporal")
    if abs(analysis['bias']) > analysis['std'] * 0.1:
        logger.warning(f"⚠️  Viés detectado: {analysis['bias']:.2f} — modelo sistematicamente {'sub' if analysis['bias'] > 0 else 'super'}estima")

    return analysis


# ──────────────────────────────────────────────
# Relatório de avaliação
# ──────────────────────────────────────────────

def generate_evaluation_report(model_name: str,
                                cv_results: pd.DataFrame,
                                baseline_results: dict,
                                residual_analysis: dict,
                                save_path: str = None) -> pd.DataFrame:
    """Gera relatório consolidado de avaliação."""
    summary = pd.DataFrame({
        'Modelo': [model_name, 'Naive', 'Seasonal Naive'],
        'RMSE'  : [
            cv_results['rmse'].mean(),
            baseline_results['naive']['rmse'],
            baseline_results['seasonal_naive']['rmse']
        ],
        'MAE'   : [
            cv_results['mae'].mean(),
            baseline_results['naive']['mae'],
            baseline_results['seasonal_naive']['mae']
        ],
        'MAPE'  : [
            cv_results['mape'].mean(),
            baseline_results['naive']['mape'],
            baseline_results['seasonal_naive']['mape']
        ],
    })

    beats_naive = summary.loc[0, 'RMSE'] < summary.loc[1, 'RMSE']
    logger.info(f"\n{'='*50}")
    logger.info(f"RELATÓRIO: {model_name}")
    logger.info(f"RMSE médio CV: {cv_results['rmse'].mean():.2f} ± {cv_results['rmse'].std():.2f}")
    logger.info(f"Bate naive: {'✅ SIM' if beats_naive else '❌ NÃO — revisar feature engineering'}")
    logger.info(f"Autocorrelação nos resíduos: {'⚠️  SIM' if residual_analysis['has_autocorr'] else '✅ Não'}")
    logger.info(f"{'='*50}\n")

    if save_path:
        summary.to_csv(save_path, index=False)

    return summary


def plot_evaluation(cv_results: pd.DataFrame, y_true: pd.Series,
                     y_pred: np.ndarray, save_dir: str = 'reports'):
    """Painel de avaliação com 4 gráficos."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. RMSE por fold
    axes[0, 0].bar(cv_results['fold'], cv_results['rmse'])
    axes[0, 0].set_title('RMSE por Fold (CV Temporal)')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('RMSE')

    # 2. Real vs previsto
    axes[0, 1].plot(y_true.values, label='Real', alpha=0.8)
    axes[0, 1].plot(y_pred, label='Previsto', alpha=0.8, linestyle='--')
    axes[0, 1].legend()
    axes[0, 1].set_title('Real vs Previsto')

    # 3. Resíduos ao longo do tempo
    residuals = y_true.values - y_pred
    axes[1, 0].plot(residuals)
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_title('Resíduos ao longo do tempo')

    # 4. Distribuição dos resíduos
    axes[1, 1].hist(residuals, bins=30, edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_title('Distribuição dos Resíduos')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'evaluation_panel.png'), dpi=150)
    logger.info(f"Painel salvo em {save_dir}/evaluation_panel.png")
