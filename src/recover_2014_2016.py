"""
================================================================================
 MÓDULO: recover_2014_2016.py
 PROJETO: ML-Forecast-Seg — Previsão de Novos Casos Judiciais (TJGO)
================================================================================
 Diagnóstico e Correção: O arquivo 2014-2016 tem 13 colunas por linha de dados
 mas o header original declara apenas 12. Isso causa deslocamento e descarte
 em massa das linhas ao usar on_bad_lines='skip'.

 Solução: Leitura com header=None + mapeamento manual + parse de data %d/%m/%y.
================================================================================
"""
import os
import glob
import pandas as pd

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

# Colunas corretas do arquivo 2014-2016 (13 campos: 12 padrão + 1 extra no VALOR_CAUSA)
COLS_2014 = [
    'NUMERO', 'DATA_RECEBIMENTO', 'PRIORIDADE', 'SEGREDO_JUSTICA',
    'SERVENTIA', 'COMARCA', 'CLASSE', 'ASSUNTOS', 'PROC_STATUS',
    'AREA', 'AREA_ACAO', 'VALOR_CAUSA', 'EXTRA'
]

# Colunas padrão dos demais arquivos
COLS_STD = [
    'NUMERO', 'DATA_RECEBIMENTO', 'PRIORIDADE', 'SEGREDO_JUSTICA',
    'SERVENTIA', 'COMARCA', 'CLASSE', 'ASSUNTOS', 'PROC_STATUS',
    'AREA', 'AREA_ACAO', 'VALOR_CAUSA'
]


def load_2014_2016():
    """Carrega e corrige o arquivo 2014-2016 com schema de 13 colunas."""
    path = os.path.join(RAW_DIR, 'dados_serie_temporal_2014-01-01_a_2016-12-31.csv')
    print(f"  📄 Lendo (modo corrigido): {os.path.basename(path)} ...")

    df = pd.read_csv(
        path,
        header=0,           # usa a 1ª linha como header mas sobrescreve com names=
        names=COLS_2014,    # forçar 13 colunas
        engine='c',
        on_bad_lines='skip',
        low_memory=False
    )

    # Remover linha de header original que virou dado
    df = df[df['NUMERO'] != 'NUMERO']

    # Parse de data com formato de 2 dígitos no ano
    df['DATA_RECEBIMENTO'] = pd.to_datetime(
        df['DATA_RECEBIMENTO'], format='%d/%m/%y', errors='coerce'
    )
    df = df.dropna(subset=['DATA_RECEBIMENTO'])

    # Descartar coluna extra
    df = df.drop(columns=['EXTRA'], errors='ignore')

    # Garantir compatibilidade de tipos com os demais arquivos
    df = df[COLS_STD]

    print(f"    ✅ Recuperados: {len(df):,} registros | "
          f"{df['DATA_RECEBIMENTO'].min().date()} → {df['DATA_RECEBIMENTO'].max().date()}")
    return df


def load_standard_files():
    """Carrega os demais arquivos CSVs com o schema padrão."""
    pattern = os.path.join(RAW_DIR, 'dados_serie_temporal_201[7-9]*.csv')
    others = sorted(
        glob.glob(os.path.join(RAW_DIR, '*.csv'))
    )
    # Excluir o arquivo 2014-2016
    others = [f for f in others if '2014' not in f]

    frames = []
    for f in others:
        print(f"  📄 Lendo: {os.path.basename(f)} ...")
        chunk = pd.read_csv(f, engine='c', on_bad_lines='skip', low_memory=False)
        chunk['DATA_RECEBIMENTO'] = pd.to_datetime(
            chunk['DATA_RECEBIMENTO'], format='%d/%m/%y', errors='coerce'
        )
        chunk = chunk.dropna(subset=['DATA_RECEBIMENTO'])
        frames.append(chunk[COLS_STD])
    return pd.concat(frames, ignore_index=True)


def consolidate_all():
    """Consolida todos os arquivos num único DataFrame limpo."""
    print("\n" + "=" * 72)
    print(" RECUPERAÇÃO + CONSOLIDAÇÃO TOTAL (2014–2024)")
    print("=" * 72)

    df_2014 = load_2014_2016()
    df_rest = load_standard_files()

    df = pd.concat([df_2014, df_rest], ignore_index=True)
    df = df.sort_values('DATA_RECEBIMENTO').reset_index(drop=True)

    # Padronizar campos de texto
    df['COMARCA'] = df['COMARCA'].str.strip().str.upper()
    df['SERVENTIA'] = df['SERVENTIA'].str.strip()

    print(f"\n  📊 Dataset consolidado final:")
    print(f"     Total de registros: {len(df):,}")
    print(f"     Período:            {df['DATA_RECEBIMENTO'].min().date()} → {df['DATA_RECEBIMENTO'].max().date()}")
    print(f"     Comarcas únicas:    {df['COMARCA'].nunique()}")
    print(f"     Serventias únicas:  {df['SERVENTIA'].nunique()}")
    print(f"     Anos cobertos:      {sorted(df['DATA_RECEBIMENTO'].dt.year.unique())}")
    return df


def aggregate_monthly(df):
    """Agrega contagem de processos por Mês × Comarca × Serventia."""
    print("\n" + "=" * 72)
    print(" AGREGAÇÃO MENSAL (COMARCA × SERVENTIA) — Dataset Completo")
    print("=" * 72)

    df['ANO_MES'] = df['DATA_RECEBIMENTO'].dt.to_period('M')
    df['ANO'] = df['DATA_RECEBIMENTO'].dt.year
    df['MES'] = df['DATA_RECEBIMENTO'].dt.month

    agg = (
        df.groupby(['ANO_MES', 'COMARCA', 'SERVENTIA'])
        .agg(
            novos_casos=('NUMERO', 'count'),
            area_predominante=('AREA', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Cível')
        )
        .reset_index()
    )
    agg['ANO_MES'] = agg['ANO_MES'].astype(str)
    agg['ANO_MES_DT'] = pd.to_datetime(agg['ANO_MES'])
    agg['ANO'] = agg['ANO_MES_DT'].dt.year
    agg['MES'] = agg['ANO_MES_DT'].dt.month

    print(f"  Pares Comarca-Serventia: {agg[['COMARCA','SERVENTIA']].drop_duplicates().shape[0]:,}")
    print(f"  Horizonte temporal:      {agg['ANO_MES'].min()} → {agg['ANO_MES'].max()}")
    print(f"  Total registros agg:     {len(agg):,}")
    return agg


def zero_fill(agg):
    """Preenche meses sem demanda com zero (tratamento de esparsidade)."""
    print("\n  🔧 Aplicando zero-fill de esparsidade...")
    all_months = sorted(agg['ANO_MES'].unique())
    all_pairs = agg[['COMARCA', 'SERVENTIA']].drop_duplicates().copy()

    months_df = pd.DataFrame({'ANO_MES': all_months, '_k': 1})
    all_pairs['_k'] = 1
    grid = months_df.merge(all_pairs, on='_k').drop('_k', axis=1)

    filled = grid.merge(agg, on=['ANO_MES', 'COMARCA', 'SERVENTIA'], how='left')
    filled['novos_casos'] = filled['novos_casos'].fillna(0).astype(int)
    filled['ANO_MES_DT'] = pd.to_datetime(filled['ANO_MES'])
    filled['ANO'] = filled['ANO_MES_DT'].dt.year
    filled['MES'] = filled['ANO_MES_DT'].dt.month

    # Propagar área predominante
    mode_area = agg.groupby('SERVENTIA')['area_predominante'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Cível'
    ).reset_index().rename(columns={'area_predominante': 'area_fill'})
    filled = filled.merge(mode_area, on='SERVENTIA', how='left')
    filled['area_predominante'] = filled['area_predominante'].fillna(filled['area_fill'])
    filled.drop('area_fill', axis=1, inplace=True)

    n_zeros = filled['novos_casos'].eq(0).sum()
    pct = n_zeros / len(filled) * 100
    print(f"  Grid total:   {len(filled):,} | Zeros imputados: {n_zeros:,} ({pct:.1f}%)")
    return filled


def feature_engineering(df):
    """Gera features temporais: lags, rolling means, calendário."""
    print("\n" + "=" * 72)
    print(" ENGENHARIA DE FEATURES")
    print("=" * 72)

    df = df.sort_values(['COMARCA', 'SERVENTIA', 'ANO_MES_DT']).reset_index(drop=True)
    g = ['COMARCA', 'SERVENTIA']

    for lag in [1, 2, 3, 6, 12]:
        df[f'lag_{lag}'] = df.groupby(g)['novos_casos'].shift(lag)
        print(f"  ✔ lag_{lag}")

    for w in [3, 6, 12]:
        df[f'rolling_mean_{w}'] = df.groupby(g)['novos_casos'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        print(f"  ✔ rolling_mean_{w}")

    df['rolling_std_3'] = df.groupby(g)['novos_casos'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std())
    print(f"  ✔ rolling_std_3")

    df['mes_do_ano'] = df['MES']
    df['trimestre'] = ((df['MES'] - 1) // 3) + 1
    df['is_recesso'] = df['MES'].isin([1, 7]).astype(int)
    df['is_pandemia'] = df['ANO'].isin([2020, 2021]).astype(int)
    df['mes_sin'] = (2 * 3.14159265 * df['MES'] / 12).apply(__import__('math').sin)
    df['mes_cos'] = (2 * 3.14159265 * df['MES'] / 12).apply(__import__('math').cos)
    df['area_civel'] = (df['area_predominante'] == 'Cível').astype(int)

    features = [c for c in df.columns if c not in [
        'COMARCA', 'SERVENTIA', 'ANO_MES', 'ANO_MES_DT',
        'novos_casos', 'area_predominante', 'ANO', 'MES'
    ]]
    print(f"\n  ✅ Total de features: {len(features)}")
    return df


def split_and_export(df):
    """Split Out-of-Time e exportação dos datasets."""
    print("\n" + "=" * 72)
    print(" SPLIT OUT-OF-TIME + EXPORTAÇÃO")
    print("=" * 72)

    feat_cols = [c for c in df.columns if c.startswith('lag_') or c.startswith('rolling_')]
    clean = df.dropna(subset=feat_cols)
    print(f"  Removidos por NaN em lags (warm-up): {len(df) - len(clean):,}")

    train = clean[clean['ANO'] <= 2023].copy()
    test = clean[clean['ANO'] == 2024].copy()

    # Verificação anti-leakage
    assert train['ANO_MES_DT'].max() < test['ANO_MES_DT'].min(), "⚠️ DATA LEAKAGE!"
    print(f"  ✅ Anti-leakage OK: treino até {train['ANO_MES_DT'].max().date()}")

    # Converter datetime para string antes de exportar
    for ds in [train, test, clean]:
        ds['ANO_MES_DT'] = ds['ANO_MES_DT'].astype(str)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(f'{PROCESSED_DIR}/train_full.csv', index=False)
    test.to_csv(f'{PROCESSED_DIR}/test_full.csv', index=False)
    clean.to_csv(f'{PROCESSED_DIR}/full_prepared_v2.csv', index=False)

    print(f"\n  ✔ train_full.csv:          {len(train):,} linhas (2014–2023)")
    print(f"  ✔ test_full.csv:           {len(test):,} linhas (2024)")
    print(f"  ✔ full_prepared_v2.csv:    {len(clean):,} linhas (completo)")


def main():
    print("=" * 72)
    print(" RECOVER + REPROCESS — Recuperação dos dados 2014-2016")
    print("=" * 72)

    df_raw = consolidate_all()
    agg = aggregate_monthly(df_raw)
    del df_raw

    filled = zero_fill(agg)
    del agg

    featured = feature_engineering(filled)
    del filled

    split_and_export(featured)

    print("\n" + "=" * 72)
    print(" ✅  PIPELINE COMPLETO — Dataset estendido 2014–2024 pronto!")
    print("=" * 72)


if __name__ == '__main__':
    main()
