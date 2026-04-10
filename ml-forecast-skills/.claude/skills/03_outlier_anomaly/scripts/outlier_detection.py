"""
src/features/outlier_detection.py
Detecção e tratamento de outliers/anomalias em séries temporais epidemiológicas.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import logging, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import TARGET_COL

logger = logging.getLogger(__name__)


def detect_rolling_zscore(series: pd.Series, window: int = 30,
                           threshold: float = 3.0) -> pd.Series:
    """
    Z-score rolling — mais robusto que z-score global para séries temporais.
    Usa janela centrada para contexto simétrico.
    """
    rolling_mean = series.rolling(window, center=True, min_periods=7).mean()
    rolling_std  = series.rolling(window, center=True, min_periods=7).std()
    z_scores = (series - rolling_mean) / (rolling_std + 1e-8)
    return z_scores.abs() > threshold


def detect_iqr_rolling(series: pd.Series, window: int = 30,
                        multiplier: float = 1.5) -> pd.Series:
    """IQR rolling — robusto a distribuições assimétricas (comum em epidemiologia)."""
    q1 = series.rolling(window, center=True, min_periods=7).quantile(0.25)
    q3 = series.rolling(window, center=True, min_periods=7).quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def detect_isolation_forest(df: pd.DataFrame, feature_cols: list,
                              contamination: float = 0.05) -> pd.Series:
    """
    Isolation Forest — detecta anomalias multivariadas.
    Útil quando outliers dependem de contexto (ex: pico em dia da semana incomum).
    contamination: proporção esperada de outliers (default 5%).
    """
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    X = df[feature_cols].fillna(df[feature_cols].median())
    predictions = iso.fit_predict(X)
    return pd.Series(predictions == -1, index=df.index)


def classify_outlier(series: pd.Series, outlier_mask: pd.Series,
                      context_window: int = 7) -> pd.Series:
    """
    Classifica outliers por tipo para decisão de tratamento:
    - 'spike': pico pontual isolado
    - 'level_shift': mudança de nível persistente
    - 'unknown': requer investigação manual
    """
    classification = pd.Series('none', index=series.index)

    for date in series[outlier_mask].index:
        # Verifica se outliers vizinhos também são anômalos (level shift)
        window_before = outlier_mask.loc[:date].tail(context_window + 1)[:-1]
        window_after  = outlier_mask.loc[date:].head(context_window + 1)[1:]

        n_neighbors_anomalous = window_before.sum() + window_after.sum()

        if n_neighbors_anomalous == 0:
            classification.loc[date] = 'spike'
        elif n_neighbors_anomalous >= context_window // 2:
            classification.loc[date] = 'level_shift'
        else:
            classification.loc[date] = 'unknown'

    return classification


def treat_outliers(df: pd.DataFrame, outlier_mask: pd.Series,
                    classification: pd.Series) -> pd.DataFrame:
    """
    Estratégia de tratamento por tipo de outlier:
    - spike: interpolar (substituir pelo contexto local)
    - level_shift: manter + adicionar feature binária (info real)
    - unknown: interpolar + logar para revisão manual
    """
    df = df.copy()
    df['is_outlier']      = outlier_mask.astype(int)
    df['outlier_type']    = classification
    df[f'{TARGET_COL}_original'] = df[TARGET_COL].copy()
    df[f'{TARGET_COL}_clean']    = df[TARGET_COL].copy()

    # Spikes: substituir por NaN e interpolar
    spike_mask = (outlier_mask) & (classification == 'spike')
    df.loc[spike_mask, f'{TARGET_COL}_clean'] = np.nan
    df[f'{TARGET_COL}_clean'] = df[f'{TARGET_COL}_clean'].interpolate(method='linear')

    # Level shifts: manter valor, adicionar indicador
    level_shift_mask = (outlier_mask) & (classification == 'level_shift')
    df['is_level_shift'] = level_shift_mask.astype(int)

    n_spikes      = spike_mask.sum()
    n_lvl_shifts  = level_shift_mask.sum()
    n_unknown     = ((outlier_mask) & (classification == 'unknown')).sum()
    logger.info(f"Outliers tratados — spikes: {n_spikes}, level_shifts: {n_lvl_shifts}, unknown: {n_unknown}")

    if n_unknown > 0:
        unknown_dates = df[outlier_mask & (classification == 'unknown')].index.tolist()
        logger.warning(f"⚠️  Datas para revisão manual: {unknown_dates}")

    return df


def full_outlier_pipeline(df: pd.DataFrame,
                           zscore_window: int = 30,
                           zscore_threshold: float = 3.0) -> pd.DataFrame:
    """Pipeline completo: detecção → classificação → tratamento."""
    series = df[TARGET_COL]

    # Combina dois detectores (união — mais conservador)
    mask_zscore = detect_rolling_zscore(series, window=zscore_window, threshold=zscore_threshold)
    mask_iqr    = detect_iqr_rolling(series, window=zscore_window)
    combined    = mask_zscore & mask_iqr   # interseção = somente onde ambos concordam

    n_detected = combined.sum()
    logger.info(f"Outliers detectados: {n_detected} ({n_detected/len(series)*100:.1f}%)")

    classification = classify_outlier(series, combined)
    df_treated = treat_outliers(df, combined, classification)
    return df_treated


def plot_outliers(df: pd.DataFrame, save_path: str = 'reports/outliers.png'):
    """Visualização de outliers detectados vs série limpa."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(df[TARGET_COL], label='Original', alpha=0.7)
    if 'is_outlier' in df.columns:
        outliers = df[df['is_outlier'] == 1]
        axes[0].scatter(outliers.index, outliers[TARGET_COL],
                        color='red', zorder=5, label='Outlier detectado', s=40)
    axes[0].legend()
    axes[0].set_title('Série Original com Outliers Detectados')

    if f'{TARGET_COL}_clean' in df.columns:
        axes[1].plot(df[f'{TARGET_COL}_clean'], label='Série Limpa', color='green', alpha=0.8)
        axes[1].plot(df[TARGET_COL], label='Original', alpha=0.3, linestyle='--')
        axes[1].legend()
        axes[1].set_title('Série após Tratamento')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Gráfico salvo em {save_path}")


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/series_clean.parquet")
    df_treated = full_outlier_pipeline(df)
    plot_outliers(df_treated)
    df_treated.to_parquet("data/processed/series_outlier_treated.parquet")
