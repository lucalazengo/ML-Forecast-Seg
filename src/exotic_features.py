"""
================================================================================
 MÓDULO: exotic_features.py
 ENGENHARIA DE FEATURES EXÓTICA para Séries Temporais Judiciais
================================================================================
 Técnicas avançadas para reduzir WMAPE (<27%):
 1. Fourier Features (captura ciclos legais complexos)
 2. Quantile Features (captura comportamento nas caudas)
 3. Detrended Components (remove drift sistemático)
 4. Local Level/Slope (dinâmica de suavização exponencial)
 5. Multiplicative Seasonal Indices (sazonalidade sofisticada)
 6. Volatility Lags (momentum da incerteza)
 7. Rate-of-Change Features (derivadas e aceleração)
 8. Cross-sectional Interactions (padrões comarca × serventia)
================================================================================
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d


def add_fourier_features(df, group_cols, target='novos_casos', periods=[12, 6, 4, 3], max_order=3):
    """
    Adiciona features Fourier para capturar múltiplos ciclos sazonais.
    IMPORTANTE: Usa índice temporal ABSOLUTO (meses desde época fixa)
    para garantir alinhamento de fase entre treino e inferência.

    Períodos típicos do judiciário:
    - 12 meses: ciclo anual completo
    - 6 meses: semestres judiciais
    - 4 meses: trimestres + variações
    - 3 meses: trimestres
    """
    print("\n  ✔ Adicionando Fourier Features...")

    df = df.sort_values(group_cols + ['ANO_MES_DT']).reset_index(drop=True)

    # Índice temporal absoluto: meses desde jan/2014 (início da série)
    # Isso garante que a fase seja idêntica entre treino e inferência
    t = (df['ANO_MES_DT'].dt.year - 2014) * 12 + (df['ANO_MES_DT'].dt.month - 1)
    t = t.values

    for period in periods:
        for order in range(1, max_order + 1):
            col_sin = f'fourier_{period}m_sin_{order}'
            col_cos = f'fourier_{period}m_cos_{order}'

            df[col_sin] = np.sin(2 * np.pi * order * t / period)
            df[col_cos] = np.cos(2 * np.pi * order * t / period)

    return df


def add_quantile_features(df, group_cols, target='novos_casos', windows=[3, 6, 12]):
    """
    Adds quantile-based rolling features (p25, p50, p75).
    Captures the distribution of cases in different percentiles.

    Útil para:
    - Detectar high/low demand periods
    - Capturar comportamento em caudas (outlier detection)
    - Volatilidade via IQR (interquartile range)
    """
    print("  ✔ Adicionando Quantile Features...")

    for window in windows:
        for quantile in [0.25, 0.5, 0.75]:
            col_name = f'rolling_q{int(quantile*100)}_{window}'
            df[col_name] = (
                df.groupby(group_cols)[target]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).quantile(quantile))
            )

        # IQR como proxy de volatilidade
        col_iqr = f'rolling_iqr_{window}'
        q75 = df.groupby(group_cols)[target].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).quantile(0.75)
        )
        q25 = df.groupby(group_cols)[target].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).quantile(0.25)
        )
        df[col_iqr] = (q75 - q25).fillna(0)

    return df


def add_detrended_features(df, group_cols, target='novos_casos', windows=[3, 6, 12]):
    """
    Adiciona features baseadas em detrending (remoção de trend local).
    O resíduo vs trend captura anomalias e desvios do padrão esperado.

    Método: Local trend removal via differencing de baixa ordem.
    IMPORTANTE: Usa shift(1) para evitar data leakage.
    """
    print("  ✔ Adicionando Detrended Features...")

    for window in windows:
        col_trend = f'trend_{window}'
        col_detrended = f'detrended_{window}'
        col_deviation = f'deviation_{window}'

        # Trend local via rolling mean COM shift(1) — somente valores passados
        df[col_trend] = (
            df.groupby(group_cols)[target]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        # Detrended = lag_1 - trend (compara último valor conhecido com a tendência)
        lag1 = df.groupby(group_cols)[target].shift(1)
        df[col_detrended] = lag1 - df[col_trend]

        # Deviation = abs(detrended), captura magnitude de anomalias
        df[col_deviation] = np.abs(df[col_detrended])

    return df


def add_local_level_slope(df, group_cols, target='novos_casos', alpha=0.3, beta=0.1):
    """
    Implementa suavização exponencial dupla (Holt) COM shift(1).
    IMPORTANTE: O level/slope em t são calculados usando dados até t-1,
    evitando data leakage. O valor em t é a "previsão" do Holt para t.

    Parâmetros:
    - alpha (0.3): peso para level update (maior = mais reativo)
    - beta (0.1): peso para slope update (menor = smoother)
    """
    print("  ✔ Adicionando Local Level/Slope (Holt Smoothing)...")

    df = df.sort_values(group_cols + ['ANO_MES_DT']).reset_index(drop=True)

    levels = []
    slopes = []

    for _, group in df.groupby(group_cols):
        y = group[target].values.astype(float)

        # Inicializar level e slope
        level = y[0] if len(y) > 0 else 0
        slope = (y[1] - y[0]) if len(y) > 1 else 0

        # Para t=0, não temos passado — usar 0
        level_series = [0.0]
        slope_series = [0.0]

        # Para t=1, usamos apenas y[0]
        if len(y) > 1:
            level_series.append(level)
            slope_series.append(slope)

        # Para t>=2, level/slope refletem dados até t-1
        for i in range(2, len(y)):
            level_prev = level
            level = alpha * y[i - 1] + (1 - alpha) * (level + slope)
            slope = beta * (level - level_prev) + (1 - beta) * slope

            level_series.append(level)
            slope_series.append(slope)

        levels.extend(level_series)
        slopes.extend(slope_series)

    df['holt_level'] = levels
    df['holt_slope'] = slopes

    return df


def add_multiplicative_seasonal_indices(df, group_cols, target='novos_casos', period=12):
    """
    Calcula índices sazonais multiplicativos por grupo.
    IMPORTANTE: Usa expanding mean com shift(1) para evitar data leakage.

    Índice = (média sazonal passada) / (média geral passada)
    """
    print("  ✔ Adicionando Multiplicative Seasonal Indices...")

    df = df.sort_values(group_cols + ['ANO_MES_DT']).reset_index(drop=True)

    # Agrupar por mês-do-ano dentro de cada grupo
    df['mes_within_year'] = df['MES'] % period

    # Calcular média sazonal usando apenas valores passados (expanding com shift)
    seasonal_means = (
        df.groupby(group_cols + ['mes_within_year'])[target]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )

    # Média geral por grupo usando apenas valores passados
    group_means = df.groupby(group_cols)[target].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Índice sazonal = (média sazonal passada) / (média geral passada)
    df['seasonal_index'] = (seasonal_means / (group_means + 1e-6)).fillna(1.0)

    # Log do índice (mais estável para multiplicativo)
    df['seasonal_index_log'] = np.log(df['seasonal_index'] + 1e-6)

    return df


def add_volatility_dynamics(df, group_cols, target='novos_casos'):
    """
    Adiciona lags da volatilidade (rolling_std).
    Captura "momentum da incerteza" - se a volatilidade tá aumentando/diminuindo.

    Útil para:
    - Detecção de períodos turbulentos
    - Estimação de confiança em previsões
    """
    print("  ✔ Adicionando Volatility Dynamics...")

    # Base: rolling_std_3 já existe, criamos seus lags
    for lag in [1, 2, 3, 6]:
        col_name = f'rolling_std_3_lag_{lag}'
        df[col_name] = df.groupby(group_cols)['rolling_std_3'].shift(lag)

    # Aceleração de volatilidade (derivada discreta)
    df['volatility_accel'] = df['rolling_std_3'] - df['rolling_std_3'].shift(1)
    df['volatility_accel'] = df.groupby(group_cols)['volatility_accel'].transform(lambda x: x.fillna(0))

    return df


def add_rate_of_change_features(df, group_cols, target='novos_casos', periods=[1, 3, 6, 12]):
    """
    Adiciona taxa de mudança (momentum) usando SOMENTE valores passados.
    roc = (y[t-1] - y[t-1-n]) / y[t-1-n]

    Captura:
    - Crescimento/decrescimento
    - Aceleração/desaceleração
    - Mudanças abruptas (early warning)
    """
    print("  ✔ Adicionando Rate-of-Change Features...")

    # Usar lag_1 como referência (último valor conhecido) em vez do valor atual
    lag1 = df.groupby(group_cols)[target].shift(1)

    for period in periods:
        col_roc = f'roc_{period}'
        col_roc_pct = f'roc_pct_{period}'

        # shift(period) a partir do lag_1 = shift(1 + period) do original
        shifted = df.groupby(group_cols)[target].shift(1 + period)

        df[col_roc] = lag1 - shifted
        df[col_roc_pct] = (df[col_roc] / (shifted + 1e-6) * 100).fillna(0)

    return df


def add_cross_sectional_features(df, group_cols=['COMARCA', 'SERVENTIA'], target='novos_casos'):
    """
    Features baseadas em padrões cross-sectional.
    IMPORTANTE: Todas usam shift(1) ou expanding mean para evitar data leakage.

    Captura:
    - Média histórica da comarca (efeito regional)
    - Desvio relativo da serventia vs comarca
    - Ranking relativo
    """
    print("  ✔ Adicionando Cross-Sectional Features...")

    # Média por comarca usando apenas valores passados (expanding mean com shift)
    df['comarca_mean'] = df.groupby('COMARCA')[target].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    df['comarca_mean_lag'] = df.groupby('COMARCA')[target].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean()
    )

    # Desvio relativo: lag_1 da serventia vs média histórica da comarca
    lag1 = df.groupby(group_cols)[target].shift(1)
    df['deviation_from_comarca'] = lag1 - df['comarca_mean']

    # Média por serventia usando apenas valores passados
    df['serventia_mean'] = df.groupby('SERVENTIA')[target].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    # Razão: serventia vs comarca (normaliza efeitos regionais)
    df['serventia_comarca_ratio'] = (df['serventia_mean'] / (df['comarca_mean'] + 1e-6)).fillna(1.0)

    return df


def add_anomaly_features(df, group_cols, target='novos_casos', threshold_std=2.5):
    """
    Detecta anomalias e cria features baseadas em outliers.
    IMPORTANTE: Usa shift(1) para evitar data leakage — detecta se o valor
    ANTERIOR era anômalo, não o valor atual.
    """
    print("  ✔ Adicionando Anomaly Features...")

    df = df.sort_values(group_cols + ['ANO_MES_DT']).reset_index(drop=True)

    # Z-score por grupo usando apenas valores passados (shift + rolling)
    def detect_anomalies(x):
        if len(x) < 2:
            return np.zeros(len(x))
        shifted = x.shift(1)
        mean = shifted.rolling(12, min_periods=1).mean()
        std = shifted.rolling(12, min_periods=1).std()
        z_score = np.abs((shifted - mean) / (std + 1e-6))
        return (z_score > threshold_std).astype(int)

    df['is_anomaly'] = (
        df.groupby(group_cols)[target]
        .transform(detect_anomalies)
    )

    # Count de anomalias recentes (últimos 6 meses, já baseado em dados passados)
    df['anomaly_count_6m'] = (
        df.groupby(group_cols)['is_anomaly']
        .transform(lambda x: x.rolling(6, min_periods=1).sum())
    )

    return df


def apply_all_exotic_features(df, group_cols=['COMARCA', 'SERVENTIA'], target='novos_casos'):
    """
    Aplica todas as técnicas de feature engineering exótica em sequência.
    """
    print("\n" + "=" * 72)
    print(" ENGENHARIA DE FEATURES EXÓTICA")
    print("=" * 72)

    df = add_fourier_features(df, group_cols, target, periods=[12, 6, 4, 3], max_order=2)
    df = add_quantile_features(df, group_cols, target, windows=[3, 6, 12])
    df = add_detrended_features(df, group_cols, target, windows=[3, 6, 12])
    df = add_local_level_slope(df, group_cols, target, alpha=0.3, beta=0.1)
    df = add_multiplicative_seasonal_indices(df, group_cols, target, period=12)
    df = add_volatility_dynamics(df, group_cols, target)
    df = add_rate_of_change_features(df, group_cols, target, periods=[1, 3, 6, 12])
    df = add_cross_sectional_features(df, group_cols, target)
    df = add_anomaly_features(df, group_cols, target, threshold_std=2.5)

    # Remover NaNs/infs residuais
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)

    new_features = [c for c in df.columns if c not in [
        'COMARCA', 'SERVENTIA', 'ANO_MES', 'ANO_MES_DT',
        target, 'area_predominante', 'ANO', 'MES', 'mes_within_year',
        'comarca_mean', 'serventia_mean', 'trend_3', 'trend_6', 'trend_12',
        'holt_level', 'holt_slope'
    ]]

    print(f"\n  ✅ Total de features exóticas criadas: {len(new_features)}")
    print(f"  → Shape final: {df.shape}")

    return df, new_features


if __name__ == '__main__':
    # Teste rápido
    test_df = pd.DataFrame({
        'COMARCA': ['Goiânia', 'Goiânia', 'Anápolis'] * 24,
        'SERVENTIA': ['1ª', '2ª', '1ª'] * 24,
        'ANO_MES': pd.date_range('2023-01', periods=72, freq='MS'),
        'novos_casos': np.random.poisson(20, 72),
        'area_predominante': ['Cível'] * 72,
    })

    test_df['ANO_MES_DT'] = test_df['ANO_MES']
    test_df['ANO'] = test_df['ANO_MES_DT'].dt.year
    test_df['MES'] = test_df['ANO_MES_DT'].dt.month
    test_df['rolling_std_3'] = test_df.groupby(['COMARCA', 'SERVENTIA'])['novos_casos'].transform(
        lambda x: x.rolling(3, min_periods=1).std()
    )

    result, feats = apply_all_exotic_features(test_df)
    print(f"\n  Features geradas: {feats[:10]}...")
