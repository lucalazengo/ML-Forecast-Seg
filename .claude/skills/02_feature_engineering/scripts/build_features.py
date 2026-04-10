"""
src/features/build_features.py
Feature engineering especialista para XGBoost/LightGBM em séries temporais.
CRÍTICO: todo shift/lag deve respeitar o horizonte de previsão.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config import TARGET_COL, DATE_COL, HORIZON, LAG_LIST, ROLLING_WINDOWS
import logging

logger = logging.getLogger(__name__)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de calendário — captura sazonalidade semanal/mensal/anual."""
    df = df.copy()
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df[DATE_COL])

    df['dia_semana']    = idx.dayofweek          # 0=segunda, 6=domingo
    df['mes']           = idx.month
    df['trimestre']     = idx.quarter
    df['semana_ano']    = idx.isocalendar().week.astype(int)
    df['dia_ano']       = idx.dayofyear
    df['is_fim_semana'] = (idx.dayofweek >= 5).astype(int)

    # Codificação cíclica para dia_semana e mes (evita descontinuidade 0→6, 1→12)
    df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
    df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
    df['mes_sin']        = np.sin(2 * np.pi * (df['mes'] - 1) / 12)
    df['mes_cos']        = np.cos(2 * np.pi * (df['mes'] - 1) / 12)

    logger.info(f"Calendar features adicionadas: {[c for c in df.columns if c not in [TARGET_COL]]}")
    return df


def add_lag_features(df: pd.DataFrame, lag_list: list = None, horizon: int = None) -> pd.DataFrame:
    """
    Lags da série alvo.
    REGRA: lag mínimo = horizon (jamais usar lag < horizon — causa leakage).
    """
    lag_list = lag_list or LAG_LIST
    horizon  = horizon  or HORIZON
    df = df.copy()

    # Garante que nenhum lag viole o horizonte
    safe_lags = [l for l in lag_list if l >= horizon]
    if len(safe_lags) < len(lag_list):
        dropped = [l for l in lag_list if l < horizon]
        logger.warning(f"⚠️  Lags {dropped} removidos — menores que horizon={horizon} (leakage)")

    for lag in safe_lags:
        df[f'lag_{lag}'] = df[TARGET_COL].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame, windows: list = None, horizon: int = None) -> pd.DataFrame:
    """
    Estatísticas rolling com shift para evitar leakage.
    O shift garante que a janela usa apenas dados disponíveis no momento da previsão.
    """
    windows = windows or ROLLING_WINDOWS
    horizon = horizon or HORIZON
    df = df.copy()

    # Série deslocada pelo horizonte — simula o que estaria disponível no momento real
    shifted = df[TARGET_COL].shift(horizon)

    for w in windows:
        df[f'rolling_mean_{w}']  = shifted.rolling(w).mean()
        df[f'rolling_std_{w}']   = shifted.rolling(w).std()
        df[f'rolling_min_{w}']   = shifted.rolling(w).min()
        df[f'rolling_max_{w}']   = shifted.rolling(w).max()
        df[f'rolling_range_{w}'] = df[f'rolling_max_{w}'] - df[f'rolling_min_{w}']

    # Coeficiente de variação (std/mean) — captura volatilidade relativa
    for w in windows:
        mean = df[f'rolling_mean_{w}']
        std  = df[f'rolling_std_{w}']
        df[f'rolling_cv_{w}'] = std / (mean + 1e-8)

    return df


def add_trend_features(df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
    """Features de tendência de longo prazo."""
    horizon = horizon or HORIZON
    df = df.copy()

    shifted = df[TARGET_COL].shift(horizon)
    df['expanding_mean'] = shifted.expanding().mean()   # média histórica acumulada
    df['expanding_std']  = shifted.expanding().std()

    # Tendência linear simples (dias desde início)
    df['trend_index'] = np.arange(len(df))

    # Mudança percentual em relação à semana anterior
    df['pct_change_7']  = shifted.pct_change(7).replace([np.inf, -np.inf], np.nan)
    df['pct_change_28'] = shifted.pct_change(28).replace([np.inf, -np.inf], np.nan)

    return df


def add_holiday_features(df: pd.DataFrame, holidays: list = None) -> pd.DataFrame:
    """
    Features de feriados/eventos especiais.
    holidays: lista de datas no formato 'YYYY-MM-DD'
    """
    df = df.copy()
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df[DATE_COL])

    if holidays:
        holiday_dates = pd.to_datetime(holidays)
        df['is_feriado'] = idx.isin(holiday_dates).astype(int)
        # Proximidade a feriados (janela de ±3 dias)
        for offset in range(1, 4):
            df[f'pre_feriado_{offset}']  = idx.isin(holiday_dates - pd.Timedelta(days=offset)).astype(int)
            df[f'pos_feriado_{offset}']  = idx.isin(holiday_dates + pd.Timedelta(days=offset)).astype(int)
    else:
        df['is_feriado'] = 0
        logger.info("Sem feriados fornecidos — feature is_feriado zerada")

    return df


def build_all_features(df: pd.DataFrame, holidays: list = None) -> pd.DataFrame:
    """Pipeline completo de feature engineering."""
    original_cols = set(df.columns)

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_holiday_features(df, holidays=holidays)

    # Remove NaN gerados pelos lags/rolling (linhas iniciais sem contexto)
    n_before = len(df)
    df = df.dropna()
    n_after  = len(df)
    logger.info(f"Feature engineering: {len(df.columns) - len(original_cols)} features criadas. "
                f"Linhas removidas por NaN: {n_before - n_after}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Retorna lista de features (exclui target e colunas de data)."""
    exclude = {TARGET_COL, DATE_COL, 'data', 'date'}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    df = pd.read_parquet("data/processed/series_clean.parquet")
    df = build_all_features(df)
    df.to_parquet("data/processed/features.parquet")
    print(f"Features salvas. Shape: {df.shape}")
    print("Colunas:", df.columns.tolist())
