"""
================================================================================
 MÓDULO: generate_eda_plots.py
 FASE CRISP-DM: 2 - Data Understanding (Compreensão dos Dados)
 PROJETO: ML-Forecast-Seg — Previsão de Novos Casos Judiciais (TJGO)
================================================================================
 Gera visualizações interativas (Plotly HTML) e tabelas CSV para compor
 o relatório técnico da EDA destinado à Diretoria do Tribunal.
================================================================================
"""
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

IMG_DIR = 'reports/images'
TBL_DIR = 'reports/tables'
DATA_DIR = 'data/raw'

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)


def load_all_data():
    """Carrega todos os CSVs brutos e consolida num único DataFrame."""
    import glob
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    frames = []
    for f in csv_files:
        print(f"  Lendo: {os.path.basename(f)} ...")
        chunk = pd.read_csv(f, engine='c', on_bad_lines='skip', low_memory=False)
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    df['DATA_RECEBIMENTO'] = pd.to_datetime(
        df['DATA_RECEBIMENTO'], errors='coerce', dayfirst=True
    )
    df = df.dropna(subset=['DATA_RECEBIMENTO'])
    df['ANO'] = df['DATA_RECEBIMENTO'].dt.year
    df['MES'] = df['DATA_RECEBIMENTO'].dt.month
    df['ANO_MES'] = df['DATA_RECEBIMENTO'].dt.to_period('M').astype(str)
    print(f"\n✅ Dataset consolidado: {df.shape[0]:,} registros | {df.shape[1]} colunas")
    return df


def table_overview(df):
    """Gera tabela-resumo com as colunas, tipos e % de nulos."""
    info = pd.DataFrame({
        'Coluna': df.columns,
        'Tipo': df.dtypes.values.astype(str),
        'Nulos': df.isnull().sum().values,
        'Nulos (%)': (df.isnull().mean() * 100).round(2).values,
        'Únicos': df.nunique().values
    })
    path = os.path.join(TBL_DIR, '01_overview_colunas.csv')
    info.to_csv(path, index=False)
    print(f"  ✔ Tabela overview salva: {path}")
    return info


def table_cardinality(df):
    """Tabela de Cardinalidade: Comarcas e Serventias por ano."""
    rows = []
    for ano, g in df.groupby('ANO'):
        rows.append({
            'Ano': int(ano),
            'Total de Casos': len(g),
            'Comarcas Únicas': g['COMARCA'].nunique(),
            'Serventias Únicas': g['SERVENTIA'].nunique(),
            'Áreas Únicas': g['AREA'].nunique() if 'AREA' in g.columns else 0
        })
    card = pd.DataFrame(rows)
    path = os.path.join(TBL_DIR, '02_cardinalidade_anual.csv')
    card.to_csv(path, index=False)
    print(f"  ✔ Tabela cardinalidade salva: {path}")
    return card


def table_top_comarcas(df, n=15):
    """Top N Comarcas por volume total."""
    top = df['COMARCA'].value_counts().head(n).reset_index()
    top.columns = ['Comarca', 'Total de Casos']
    top['% do Total'] = (top['Total de Casos'] / len(df) * 100).round(2)
    path = os.path.join(TBL_DIR, '03_top_comarcas.csv')
    top.to_csv(path, index=False)
    print(f"  ✔ Tabela top comarcas salva: {path}")
    return top


def table_top_serventias(df, n=15):
    """Top N Serventias por volume total."""
    top = df['SERVENTIA'].value_counts().head(n).reset_index()
    top.columns = ['Serventia', 'Total de Casos']
    top['% do Total'] = (top['Total de Casos'] / len(df) * 100).round(2)
    path = os.path.join(TBL_DIR, '04_top_serventias.csv')
    top.to_csv(path, index=False)
    print(f"  ✔ Tabela top serventias salva: {path}")
    return top


def table_area_dist(df):
    """Distribuição por Área processual."""
    dist = df['AREA'].value_counts().reset_index()
    dist.columns = ['Área', 'Total de Casos']
    dist['% do Total'] = (dist['Total de Casos'] / len(df) * 100).round(2)
    path = os.path.join(TBL_DIR, '05_distribuicao_area.csv')
    dist.to_csv(path, index=False)
    print(f"  ✔ Tabela distribuição área salva: {path}")
    return dist


def plot_evolucao_mensal(df):
    """Gráfico interativo: Evolução mensal de novos casos (todos os anos)."""
    monthly = df.groupby('ANO_MES').size().reset_index(name='Casos')
    monthly = monthly.sort_values('ANO_MES')

    fig = px.line(
        monthly, x='ANO_MES', y='Casos',
        title='<b>Evolução Mensal de Novos Casos — TJGO (2014–2024)</b>',
        labels={'ANO_MES': 'Mês/Ano', 'Casos': 'Total de Casos Recebidos'},
    )
    fig.update_traces(
        line=dict(color='#1a73e8', width=2),
        mode='lines'
    )
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=13),
        xaxis=dict(tickangle=-45, dtick=6),
        height=500,
    )
    path = os.path.join(IMG_DIR, '01_evolucao_mensal.html')
    fig.write_html(path)
    print(f"  ✔ Gráfico evolução mensal salvo: {path}")
    return monthly


def plot_evolucao_anual(df):
    """Gráfico de barras: Volume totalizado por ano."""
    yearly = df.groupby('ANO').size().reset_index(name='Casos')

    fig = px.bar(
        yearly, x='ANO', y='Casos',
        title='<b>Volume Anual de Novos Casos — TJGO</b>',
        labels={'ANO': 'Ano', 'Casos': 'Total de Casos'},
        text='Casos',
        color='Casos',
        color_continuous_scale='Blues',
    )
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=13),
        showlegend=False,
        height=450,
    )
    path = os.path.join(IMG_DIR, '02_volume_anual.html')
    fig.write_html(path)
    print(f"  ✔ Gráfico volume anual salvo: {path}")


def plot_top_comarcas(df, n=15):
    """Barras horizontais: Top N comarcas."""
    top = df['COMARCA'].value_counts().head(n).reset_index()
    top.columns = ['Comarca', 'Casos']
    top = top.sort_values('Casos')

    fig = px.bar(
        top, y='Comarca', x='Casos', orientation='h',
        title=f'<b>Top {n} Comarcas por Volume de Casos (2014–2024)</b>',
        labels={'Casos': 'Total de Casos', 'Comarca': ''},
        color='Casos', color_continuous_scale='Viridis',
        text='Casos',
    )
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=12),
        height=500, showlegend=False,
    )
    path = os.path.join(IMG_DIR, '03_top_comarcas.html')
    fig.write_html(path)
    print(f"  ✔ Gráfico top comarcas salvo: {path}")


def plot_top_serventias(df, n=15):
    """Barras horizontais: Top N serventias."""
    top = df['SERVENTIA'].value_counts().head(n).reset_index()
    top.columns = ['Serventia', 'Casos']
    top = top.sort_values('Casos')

    fig = px.bar(
        top, y='Serventia', x='Casos', orientation='h',
        title=f'<b>Top {n} Serventias por Volume de Casos (2014–2024)</b>',
        labels={'Casos': 'Total de Casos', 'Serventia': ''},
        color='Casos', color_continuous_scale='Teal',
        text='Casos',
    )
    fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=11),
        height=600, showlegend=False,
    )
    path = os.path.join(IMG_DIR, '04_top_serventias.html')
    fig.write_html(path)
    print(f"  ✔ Gráfico top serventias salvo: {path}")


def plot_area_pizza(df):
    """Pizza/Sunburst: Distribuição por Área processual."""
    dist = df['AREA'].value_counts().reset_index()
    dist.columns = ['Área', 'Casos']

    fig = px.pie(
        dist, names='Área', values='Casos',
        title='<b>Distribuição de Casos por Área Processual</b>',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=13),
        height=500,
    )
    path = os.path.join(IMG_DIR, '05_distribuicao_area.html')
    fig.write_html(path)
    print(f"  ✔ Gráfico distribuição área salvo: {path}")


def plot_heatmap_mensal(df):
    """Heatmap: Casos por Mês x Ano."""
    pivot = df.groupby(['ANO', 'MES']).size().reset_index(name='Casos')
    pivot_table = pivot.pivot(index='ANO', columns='MES', values='Casos').fillna(0)

    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=meses[:pivot_table.shape[1]],
        y=[str(y) for y in pivot_table.index],
        colorscale='YlOrRd',
        text=pivot_table.values.astype(int),
        texttemplate='%{text:,}',
        hovertemplate='Ano: %{y}<br>Mês: %{x}<br>Casos: %{z:,}<extra></extra>',
    ))
    fig.update_layout(
        title='<b>Mapa de Calor — Novos Casos por Mês/Ano</b>',
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=12),
        height=450,
        xaxis_title='Mês',
        yaxis_title='Ano',
    )
    path = os.path.join(IMG_DIR, '06_heatmap_mensal.html')
    fig.write_html(path)
    print(f"  ✔ Heatmap mensal salvo: {path}")


def plot_boxplot_comarcas_top(df, n=10):
    """Boxplot: Distribuição mensal das top N comarcas."""
    top_comarcas = df['COMARCA'].value_counts().head(n).index.tolist()
    sub = df[df['COMARCA'].isin(top_comarcas)].copy()
    monthly = sub.groupby(['ANO_MES', 'COMARCA']).size().reset_index(name='Casos')

    fig = px.box(
        monthly, x='COMARCA', y='Casos',
        title=f'<b>Variabilidade Mensal — Top {n} Comarcas</b>',
        labels={'COMARCA': 'Comarca', 'Casos': 'Casos / Mês'},
        color='COMARCA',
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=12),
        showlegend=False,
        height=500,
        xaxis_tickangle=-45,
    )
    path = os.path.join(IMG_DIR, '07_boxplot_top_comarcas.html')
    fig.write_html(path)
    print(f"  ✔ Boxplot top comarcas salvo: {path}")


def generate_summary_json(df, card_df):
    """Gera um JSON com métricas resumidas para incorporação no relatório."""
    summary = {
        'total_registros': int(df.shape[0]),
        'total_colunas': int(df.shape[1]),
        'periodo_inicio': str(df['DATA_RECEBIMENTO'].min().date()),
        'periodo_fim': str(df['DATA_RECEBIMENTO'].max().date()),
        'comarcas_unicas_total': int(df['COMARCA'].nunique()),
        'serventias_unicas_total': int(df['SERVENTIA'].nunique()),
        'areas_unicas': int(df['AREA'].nunique()) if 'AREA' in df.columns else 0,
        'nulos_valor_causa_pct': round(df['VALOR_CAUSA'].isnull().mean() * 100, 2),
        'cardinalidade_anual': card_df.to_dict(orient='records'),
    }
    path = os.path.join(TBL_DIR, '00_summary.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  ✔ JSON de resumo salvo: {path}")
    return summary


def main():
    print("=" * 72)
    print(" CRISP-DM | Fase 2 — Data Understanding | Geração de Relatório EDA")
    print("=" * 72)

    print("\n📂 [1/4] Carregando dados brutos...")
    df = load_all_data()

    print("\n📊 [2/4] Gerando tabelas analíticas...")
    tbl_overview = table_overview(df)
    tbl_card = table_cardinality(df)
    tbl_comarcas = table_top_comarcas(df)
    tbl_serventias = table_top_serventias(df)
    tbl_area = table_area_dist(df)

    print("\n📈 [3/4] Gerando gráficos interativos (Plotly HTML)...")
    plot_evolucao_mensal(df)
    plot_evolucao_anual(df)
    plot_top_comarcas(df)
    plot_top_serventias(df)
    plot_area_pizza(df)
    plot_heatmap_mensal(df)
    plot_boxplot_comarcas_top(df)

    print("\n📋 [4/4] Gerando resumo consolidado...")
    summary = generate_summary_json(df, tbl_card)

    print("\n" + "=" * 72)
    print(" ✅  EDA CONCLUÍDA COM SUCESSO")
    print(f"    Registros processados: {summary['total_registros']:,}")
    print(f"    Período: {summary['periodo_inicio']} a {summary['periodo_fim']}")
    print(f"    Comarcas: {summary['comarcas_unicas_total']} | Serventias: {summary['serventias_unicas_total']}")
    print(f"    Gráficos em: reports/images/")
    print(f"    Tabelas em:  reports/tables/")
    print("=" * 72)


if __name__ == '__main__':
    main()
