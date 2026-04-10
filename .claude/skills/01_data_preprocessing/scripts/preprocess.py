"""
src/data/preprocess.py
Pré-processamento de séries temporais para ML-Forecast-Seg.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import TARGET_COL, DATE_COL, FREQ

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_validate(filepath: str) -> pd.DataFrame:
    """Carrega e valida schema mínimo do dataframe."""
    df = pd.read_csv(filepath, parse_dates=[DATE_COL])
    assert DATE_COL in df.columns, f"Coluna '{DATE_COL}' não encontrada"
    assert TARGET_COL in df.columns, f"Coluna '{TARGET_COL}' não encontrada"
    logger.info(f"Dados carregados: {df.shape[0]} linhas, {df[DATE_COL].min()} a {df[DATE_COL].max()}")
    return df


def fix_temporal_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """Garante frequência constante e ordena por data."""
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    df = df.asfreq(FREQ)  # insere NaN onde há gaps

    n_gaps = df[TARGET_COL].isna().sum()
    if n_gaps > 0:
        logger.warning(f"⚠️  {n_gaps} gaps detectados na série — serão interpolados")
    return df


def handle_missing_values(df: pd.DataFrame, max_gap_linear: int = 3) -> pd.DataFrame:
    """
    Estratégia de imputação por tamanho do gap:
    - Gap <= max_gap_linear: interpolação linear
    - Gap > max_gap_linear: forward fill (preserva nível)
    NUNCA usa fillna(mean) — destrói estrutura temporal.
    """
    # Identifica tamanho de cada gap
    is_null = df[TARGET_COL].isna()
    gap_groups = (is_null != is_null.shift()).cumsum()

    for gid in gap_groups[is_null].unique():
        gap_size = (gap_groups == gid).sum()
        if gap_size > max_gap_linear:
            logger.warning(f"Gap longo ({gap_size} períodos) — usando forward fill")

    # Interpolação linear para gaps curtos
    df[TARGET_COL] = df[TARGET_COL].interpolate(method='linear', limit=max_gap_linear)
    # Forward fill para o restante
    df[TARGET_COL] = df[TARGET_COL].ffill()

    assert df[TARGET_COL].isna().sum() == 0, "Ainda há NaN após imputação"
    return df


def check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """Teste ADF para estacionariedade."""
    result = adfuller(series.dropna(), autolag='AIC')
    is_stationary = result[1] < significance
    logger.info(f"ADF p-value: {result[1]:.4f} — {'Estacionária ✓' if is_stationary else 'NÃO estacionária ✗'}")
    return {
        'statistic': result[0],
        'p_value': result[1],
        'is_stationary': is_stationary,
        'critical_values': result[4]
    }


def apply_transformation(df: pd.DataFrame, method: str = 'none') -> tuple:
    """
    Aplica transformação para estacionarizar se necessário.
    Retorna (df_transformado, função_inversa).
    """
    if method == 'log':
        df[TARGET_COL] = np.log1p(df[TARGET_COL])
        inverse_fn = np.expm1
        logger.info("Transformação log1p aplicada")
    elif method == 'diff':
        original_first = df[TARGET_COL].iloc[0]
        df[TARGET_COL] = df[TARGET_COL].diff().dropna()
        inverse_fn = lambda x: x.cumsum() + original_first  # noqa
        logger.info("Diferenciação de ordem 1 aplicada")
    else:
        inverse_fn = lambda x: x  # noqa
    return df, inverse_fn


def preprocess_pipeline(filepath: str, transformation: str = 'none') -> tuple:
    """Pipeline completo de pré-processamento."""
    df = load_and_validate(filepath)
    df = fix_temporal_integrity(df)
    df = handle_missing_values(df)

    stats = check_stationarity(df[TARGET_COL])
    if not stats['is_stationary'] and transformation == 'none':
        logger.warning("Série não estacionária. Considere transformation='log' ou 'diff'")

    df, inverse_fn = apply_transformation(df, method=transformation)
    logger.info(f"Pré-processamento concluído. Shape final: {df.shape}")
    return df, inverse_fn


if __name__ == "__main__":
    df, inv = preprocess_pipeline("data/raw/casos.csv")
    df.to_parquet("data/processed/series_clean.parquet")
    print(df.tail())
