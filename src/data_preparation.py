"""
================================================================================
 MÓDULO: data_preparation.py
 FASE CRISP-DM: 3 - Data Preparation (Preparação dos Dados)
 PROJETO: ML-Forecast-Seg — Previsão de Novos Casos Judiciais (TJGO)
================================================================================
 Motor de Agregação e Preparação dos Dados para modelagem de Séries Temporais.
 Granularidade: MENSAL | Agrupadores: COMARCA × SERVENTIA
 
 Etapas:
   3.1 — Consolidação de todos os CSVs brutos
   3.2 — Agregação mensal por Comarca × Serventia (contagem de processos)
   3.3 — Preenchimento de esparsidade (meses com 0 casos)
   3.4 — Engenharia de Features temporais (lags, rolling, calendário)
   3.5 — Split temporal Out-of-Time (treino 2017-2023 / teste 2024)
   3.6 — Exportação dos datasets finais
================================================================================
"""
import os
import glob
import json
import pandas as pd
import numpy as np

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'
REPORT_DIR = 'reports/tables'

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

TRAIN_END_YEAR = 2024
TEST_YEAR = 2025


# Colunas corretas do arquivo 2014-2016 (13 campos: 12 padrão + 1 extra no VALOR_CAUSA)
_COLS_2014 = [
    'NUMERO', 'DATA_RECEBIMENTO', 'PRIORIDADE', 'SEGREDO_JUSTICA',
    'SERVENTIA', 'COMARCA', 'CLASSE', 'ASSUNTOS', 'PROC_STATUS',
    'AREA', 'AREA_ACAO', 'VALOR_CAUSA', 'EXTRA'
]
_COLS_STD = [
    'NUMERO', 'DATA_RECEBIMENTO', 'PRIORIDADE', 'SEGREDO_JUSTICA',
    'SERVENTIA', 'COMARCA', 'CLASSE', 'ASSUNTOS', 'PROC_STATUS',
    'AREA', 'AREA_ACAO', 'VALOR_CAUSA'
]


def _load_2014_2016(path):
    """Carrega o arquivo 2014-2016 com schema de 13 colunas e data %d/%m/%y."""
    df = pd.read_csv(
        path, header=0, names=_COLS_2014,
        engine='c', on_bad_lines='skip', low_memory=False
    )
    df = df[df['NUMERO'] != 'NUMERO']  # remove header duplicado
    df['DATA_RECEBIMENTO'] = pd.to_datetime(
        df['DATA_RECEBIMENTO'], format='%d/%m/%y', errors='coerce'
    )
    df = df.dropna(subset=['DATA_RECEBIMENTO'])
    df = df.drop(columns=['EXTRA'], errors='ignore')
    return df[_COLS_STD]


def _load_standard(path):
    """Carrega um CSV com o schema padrão (12 colunas, data dd/mm/yy)."""
    df = pd.read_csv(path, engine='c', on_bad_lines='skip', low_memory=False)
    df['DATA_RECEBIMENTO'] = pd.to_datetime(
        df['DATA_RECEBIMENTO'], format='%d/%m/%y', errors='coerce'
    )
    df = df.dropna(subset=['DATA_RECEBIMENTO'])
    return df[_COLS_STD]


def _load_2025(path):
    """Carrega o arquivo 2025 com o schema novo."""
    df = pd.read_csv(path, engine='c', on_bad_lines='skip', low_memory=False)
    # Renomear as colunas para o schema padrao onde possivel
    df = df.rename(columns={
        'DATA_DISTRIBUICAO': 'DATA_RECEBIMENTO',
        'CODG_CLASSE': 'CLASSE',
        'CODG_ASSUNTOS': 'ASSUNTOS',
        'NOME_AREA_ACAO': 'AREA'
    })
    
    df['DATA_RECEBIMENTO'] = pd.to_datetime(
        df['DATA_RECEBIMENTO'], format='%d/%m/%y', errors='coerce'
    )
    df = df.dropna(subset=['DATA_RECEBIMENTO'])
    
    # Preencher colunas que faltam do schema padrao com valores nulos/dummies
    df['PROC_STATUS'] = 'N/A'
    df['AREA_ACAO'] = df['AREA'] # Usado como equivalente
    df['VALOR_CAUSA'] = np.nan
    
    return df[_COLS_STD]


def step_31_consolidate():
    """3.1 — Consolida todos os CSVs brutos num único DataFrame (2014–2025)."""
    print("\n" + "=" * 72)
    print(" ETAPA 3.1 — Consolidação dos CSVs Brutos (2014–2025)")
    print("=" * 72)

    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.csv')))
    frames = []
    for f in csv_files:
        print(f"  📄 Lendo: {os.path.basename(f)} ...")
        if '2014' in os.path.basename(f):
            chunk = _load_2014_2016(f)
            print(f"     ↳ Schema corrigido (13 cols, %d/%m/%y): {len(chunk):,} registros")
        elif '2025' in os.path.basename(f):
            chunk = _load_2025(f)
            print(f"     ↳ Schema 2025: {len(chunk):,} registros")
        else:
            chunk = _load_standard(f)
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values('DATA_RECEBIMENTO').reset_index(drop=True)

    # Limpar campos essenciais
    df = df.dropna(subset=['COMARCA', 'SERVENTIA'])
    df['COMARCA'] = df['COMARCA'].str.strip().str.upper()
    df['SERVENTIA'] = df['SERVENTIA'].str.strip()

    print(f"\n  ✅ Dataset consolidado: {len(df):,} registros válidos")
    print(f"     Período: {df['DATA_RECEBIMENTO'].min().date()} → {df['DATA_RECEBIMENTO'].max().date()}")
    print(f"     Anos cobertos: {sorted(df['DATA_RECEBIMENTO'].dt.year.unique())}")
    return df


def step_32_aggregate(df):
    """3.2 — Agrega contagem de processos por Mês × Comarca × Serventia."""
    print("\n" + "=" * 72)
    print(" ETAPA 3.2 — Agregação Mensal (Comarca × Serventia)")
    print("=" * 72)

    df['ANO_MES'] = df['DATA_RECEBIMENTO'].dt.to_period('M')

    agg = (
        df.groupby(['ANO_MES', 'COMARCA', 'SERVENTIA'])
        .agg(
            novos_casos=('NUMERO', 'count'),
            area_predominante=('AREA', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A')
        )
        .reset_index()
    )

    agg['ANO_MES'] = agg['ANO_MES'].astype(str)
    agg['ANO_MES_DT'] = pd.to_datetime(agg['ANO_MES'])
    agg['ANO'] = agg['ANO_MES_DT'].dt.year
    agg['MES'] = agg['ANO_MES_DT'].dt.month

    print(f"  → Séries temporais geradas: {agg[['COMARCA', 'SERVENTIA']].drop_duplicates().shape[0]:,} pares únicos")
    print(f"  → Horizonte temporal: {agg['ANO_MES'].min()} a {agg['ANO_MES'].max()}")
    print(f"  → Total de registros agregados: {len(agg):,}")
    print(f"  ✅ Agregação mensal concluída")
    return agg


def step_33_fill_sparse(agg):
    """3.3 — Preenche combinações Comarca × Serventia × Mês ausentes com 0."""
    print("\n" + "=" * 72)
    print(" ETAPA 3.3 — Preenchimento de Esparsidade (Zero-Fill)")
    print("=" * 72)

    all_months = sorted(agg['ANO_MES'].unique())
    all_pairs = agg[['COMARCA', 'SERVENTIA']].drop_duplicates()

    print(f"  → Meses no horizonte: {len(all_months)}")
    print(f"  → Pares Comarca-Serventia: {len(all_pairs):,}")
    print(f"  → Grid teórico (meses × pares): {len(all_months) * len(all_pairs):,}")

    # Criar grid completo
    months_df = pd.DataFrame({'ANO_MES': all_months})
    months_df['_key'] = 1
    all_pairs['_key'] = 1
    full_grid = months_df.merge(all_pairs, on='_key').drop('_key', axis=1)

    n_before = len(agg)
    filled = full_grid.merge(
        agg, on=['ANO_MES', 'COMARCA', 'SERVENTIA'], how='left'
    )
    filled['novos_casos'] = filled['novos_casos'].fillna(0).astype(int)
    filled['ANO_MES_DT'] = pd.to_datetime(filled['ANO_MES'])
    filled['ANO'] = filled['ANO_MES_DT'].dt.year
    filled['MES'] = filled['ANO_MES_DT'].dt.month

    # Propagar area_predominante por serventia
    mode_area = agg.groupby('SERVENTIA')['area_predominante'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Cível'
    ).reset_index()
    mode_area.columns = ['SERVENTIA', 'area_fill']
    filled = filled.merge(mode_area, on='SERVENTIA', how='left')
    filled['area_predominante'] = filled['area_predominante'].fillna(filled['area_fill'])
    filled.drop('area_fill', axis=1, inplace=True)

    n_zeros = len(filled) - n_before
    pct_sparse = (n_zeros / len(filled)) * 100

    print(f"  → Registros após zero-fill: {len(filled):,}")
    print(f"  → Zeros imputados (meses sem demanda): {n_zeros:,} ({pct_sparse:.1f}%)")
    print(f"  ✅ Esparsidade tratada com sucesso")
    return filled


def step_34_feature_engineering(filled):
    """3.4 — Engenharia de Features temporais (lags, rolling, calendário)."""
    print("\n" + "=" * 72)
    print(" ETAPA 3.4 — Engenharia de Features Temporais")
    print("=" * 72)

    filled = filled.sort_values(['COMARCA', 'SERVENTIA', 'ANO_MES_DT']).reset_index(drop=True)

    group_cols = ['COMARCA', 'SERVENTIA']

    # Lags
    for lag in [1, 2, 3, 6, 12]:
        col_name = f'lag_{lag}'
        filled[col_name] = filled.groupby(group_cols)['novos_casos'].shift(lag)
        print(f"  ✔ Feature criada: {col_name}")

    # Rolling means
    for window in [3, 6, 12]:
        col_name = f'rolling_mean_{window}'
        filled[col_name] = (
            filled.groupby(group_cols)['novos_casos']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        print(f"  ✔ Feature criada: {col_name}")

    # Rolling std (volatilidade)
    filled['rolling_std_3'] = (
        filled.groupby(group_cols)['novos_casos']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
    )
    print(f"  ✔ Feature criada: rolling_std_3")

    # Features calendário
    filled['mes_do_ano'] = filled['MES']
    filled['trimestre'] = ((filled['MES'] - 1) // 3) + 1
    filled['is_recesso'] = filled['MES'].isin([1, 7]).astype(int)  # Jan e Jul = recesso forense
    filled['is_pandemia'] = ((filled['ANO'] == 2020) | (filled['ANO'] == 2021)).astype(int)

    # Encoding cíclico para mês
    filled['mes_sin'] = np.sin(2 * np.pi * filled['MES'] / 12)
    filled['mes_cos'] = np.cos(2 * np.pi * filled['MES'] / 12)

    print(f"  ✔ Features calendário criadas: mes_do_ano, trimestre, is_recesso, is_pandemia, mes_sin, mes_cos")

    # Encoding de área
    filled['area_civel'] = (filled['area_predominante'] == 'Cível').astype(int)

    total_features = [c for c in filled.columns if c not in ['COMARCA', 'SERVENTIA', 'ANO_MES', 'ANO_MES_DT', 'novos_casos', 'area_predominante', 'ANO', 'MES']]
    print(f"\n  ✅ Total de features geradas: {len(total_features)}")
    print(f"  → Lista: {total_features}")
    return filled


def step_35_split(filled):
    """3.5 — Split temporal Out-of-Time: treino (2017-2024) vs teste (2025)."""
    print("\n" + "=" * 72)
    print(" ETAPA 3.5 — Split Out-of-Time (Treino / Teste)")
    print("=" * 72)

    # Remover linhas com NaN nos lags (primeiros 12 meses de cada série)
    feature_cols = [c for c in filled.columns if c.startswith('lag_') or c.startswith('rolling_')]
    n_before = len(filled)
    clean = filled.dropna(subset=feature_cols)
    n_dropped = n_before - len(clean)
    print(f"  → Registros removidos por NaN em lags (warm-up period): {n_dropped:,}")

    train = clean[clean['ANO'] <= TRAIN_END_YEAR].copy()
    test = clean[clean['ANO'] == TEST_YEAR].copy()

    train_start_year = train['ANO'].min()
    print(f"  → Treino: {len(train):,} registros ({train_start_year}–{TRAIN_END_YEAR})")
    print(f"  → Teste:  {len(test):,} registros ({TEST_YEAR})")
    print(f"  → Meses no treino: {train['ANO_MES'].nunique()}")
    print(f"  → Meses no teste:  {test['ANO_MES'].nunique()}")

    # Verificação anti-leakage
    train_max_date = train['ANO_MES_DT'].max()
    test_min_date = test['ANO_MES_DT'].min()
    assert train_max_date < test_min_date, "⚠️ DATA LEAKAGE DETECTADO!"
    print(f"  ✅ Verificação anti-leakage OK: treino até {train_max_date.date()}, teste desde {test_min_date.date()}")

    return train, test, clean


def step_36_export(train, test, full):
    """3.6 — Exporta datasets processados."""
    print("\n" + "=" * 72)
    print(" ETAPA 3.6 — Exportação dos Datasets Processados")
    print("=" * 72)

    # Converter datetime para string para salvar em CSV
    for ds in [train, test, full]:
        ds['ANO_MES_DT'] = ds['ANO_MES_DT'].astype(str)

    train_path = os.path.join(PROCESSED_DIR, 'train_full.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test_full.csv')
    full_path = os.path.join(PROCESSED_DIR, 'full_prepared_v2.csv')

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    full.to_csv(full_path, index=False)

    print(f"  ✔ Treino salvo: {train_path} ({len(train):,} linhas)")
    print(f"  ✔ Teste salvo:  {test_path} ({len(test):,} linhas)")
    print(f"  ✔ Completo salvo: {full_path} ({len(full):,} linhas)")

    # Gerar resumo da preparação
    summary = {
        'etapa': 'Data Preparation (Fase 3 CRISP-DM)',
        'granularidade': 'Mensal',
        'agrupadores': ['COMARCA', 'SERVENTIA'],
        'target': 'novos_casos',
        'split': {
            'treino': f"{train['ANO'].min()}-{TRAIN_END_YEAR}",
            'teste': str(TEST_YEAR),
            'treino_registros': len(train),
            'teste_registros': len(test),
        },
        'features': [c for c in train.columns if c not in [
            'COMARCA', 'SERVENTIA', 'ANO_MES', 'ANO_MES_DT',
            'novos_casos', 'area_predominante', 'ANO', 'MES'
        ]],
        'feature_count': len([c for c in train.columns if c not in [
            'COMARCA', 'SERVENTIA', 'ANO_MES', 'ANO_MES_DT',
            'novos_casos', 'area_predominante', 'ANO', 'MES'
        ]]),
    }
    summary_path = os.path.join(REPORT_DIR, '06_data_prep_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Resumo salvo: {summary_path}")

    print(f"\n  ✅ Exportação concluída com sucesso")
    return summary


def main():
    print("=" * 72)
    print(" CRISP-DM | Fase 3 — Data Preparation | Motor de Agregação")
    print(" Granularidade: MENSAL | Target: novos_casos")
    print("=" * 72)

    df = step_31_consolidate()
    agg = step_32_aggregate(df)

    del df  # Liberar memória

    filled = step_33_fill_sparse(agg)

    del agg

    featured = step_34_feature_engineering(filled)

    del filled

    train, test, full = step_35_split(featured)
    summary = step_36_export(train, test, full)

    print("\n" + "=" * 72)
    print(" ✅  PREPARAÇÃO DE DADOS CONCLUÍDA COM SUCESSO")
    print(f"    Target: novos_casos (contagem mensal)")
    print(f"    Features: {summary['feature_count']}")
    print(f"    Treino: {summary['split']['treino']} ({summary['split']['treino_registros']:,} registros)")
    print(f"    Teste:  {summary['split']['teste']} ({summary['split']['teste_registros']:,} registros)")
    print(f"    Saída em: {PROCESSED_DIR}/")
    print("=" * 72)


if __name__ == '__main__':
    main()
