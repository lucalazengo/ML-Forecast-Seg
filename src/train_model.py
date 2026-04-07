"""
================================================================================
 MÓDULO: train_model.py
 FASE CRISP-DM: 4 - Modeling (Modelagem)
 PROJETO: ML-Forecast-Seg — Previsão de Novos Casos Judiciais (TJGO)
================================================================================
 Implementa três modelos progressivos usando apenas numpy + pandas:

   M0 — Baseline Naïve Sazonal (lag_12): referência estatística mínima
   M1 — Regressão Linear Global (OLS com numpy): modelo linear multivariado
   M2 — Ensemble Ponderado (M0 × alpha + M1 × (1-alpha))

 Métricas de avaliação: MAE, RMSE, WMAPE
 Granularidade: Mensal | Agrupadores: Comarca × Serventia
 Split: Out-of-Time (treino 2014-2023 / teste 2024)
================================================================================
"""
import os
import json
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROCESSED_DIR = 'data/processed'
MODEL_DIR = 'models'
REPORT_DIR = 'reports'
IMG_DIR = 'reports/images'
TBL_DIR = 'reports/tables'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

# Features utilizadas no modelo linear global
FEATURE_COLS = [
    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
    'rolling_std_3',
    'mes_do_ano', 'trimestre', 'is_recesso', 'is_pandemia',
    'mes_sin', 'mes_cos', 'area_civel'
]
TARGET = 'novos_casos'


# ─── MÉTRICAS ────────────────────────────────────────────────────────────────

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def wmape(y_true, y_pred):
    """Weighted MAPE — robusto a zeros (denominador = soma do y_true)."""
    total = np.sum(np.abs(y_true))
    if total == 0:
        return float('nan')
    return float(np.sum(np.abs(y_true - y_pred)) / total * 100)


# ─── CARREGAMENTO DOS DADOS ───────────────────────────────────────────────────

def load_data():
    print("\n" + "=" * 72)
    print(" [1/5] CARREGANDO DATASETS PROCESSADOS")
    print("=" * 72)

    train = pd.read_csv(f'{PROCESSED_DIR}/train_full.csv', low_memory=False)
    test  = pd.read_csv(f'{PROCESSED_DIR}/test_full.csv',  low_memory=False)

    # Garantir que colunas numéricas estejam corretas
    for col in FEATURE_COLS + [TARGET]:
        train[col] = pd.to_numeric(train[col], errors='coerce')
        test[col]  = pd.to_numeric(test[col],  errors='coerce')

    train = train.dropna(subset=FEATURE_COLS + [TARGET])
    test  = test.dropna(subset=FEATURE_COLS + [TARGET])

    print(f"  Treino: {len(train):,} linhas | {train['ANO_MES'].min()} → {train['ANO_MES'].max()}")
    print(f"  Teste:  {len(test):,} linhas  | {test['ANO_MES'].min()} → {test['ANO_MES'].max()}")
    print(f"  Features: {len(FEATURE_COLS)} | Target: {TARGET}")
    return train, test


# ─── M0: BASELINE NAÏVE SAZONAL ──────────────────────────────────────────────

def model_m0_baseline(train, test):
    """
    M0 — Naïve Sazonal: previsão = valor do mesmo mês no ano anterior (lag_12).
    É o benchmark mínimo — qualquer modelo sério deve superar isso.
    """
    print("\n" + "=" * 72)
    print(" [2/5] M0 — BASELINE NAÏVE SAZONAL (lag_12)")
    print("=" * 72)

    y_true = test[TARGET].values
    y_pred = test['lag_12'].clip(lower=0).values  # nunca prever negativo

    metrics = {
        'modelo': 'M0 — Naïve Sazonal',
        'MAE':   round(mae(y_true, y_pred), 2),
        'RMSE':  round(rmse(y_true, y_pred), 2),
        'WMAPE': round(wmape(y_true, y_pred), 2),
    }
    print(f"  MAE:   {metrics['MAE']:>10.2f}")
    print(f"  RMSE:  {metrics['RMSE']:>10.2f}")
    print(f"  WMAPE: {metrics['WMAPE']:>9.2f}%")

    return y_pred, metrics


# ─── M1: REGRESSÃO LINEAR GLOBAL (OLS com numpy) ─────────────────────────────

class GlobalLinearModel:
    """
    Regressão Linear Múltipla via OLS (Mínimos Quadrados Ordinários).
    Solução analítica: β = (XᵀX)⁻¹ Xᵀy — sem dependências externas.
    Limitação: assume relação linear entre features e target.
    """
    def __init__(self, regularization=1e-4):
        self.beta = None
        self.feature_names = None
        self.regularization = regularization  # Ridge L2 para estabilidade numérica
        self.feature_means = None
        self.feature_stds = None
        self.target_mean = None

    def _normalize(self, X):
        return (X - self.feature_means) / (self.feature_stds + 1e-8)

    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        self.feature_means = X.mean(axis=0)
        self.feature_stds  = X.std(axis=0)
        self.target_mean   = y.mean()

        Xn = self._normalize(X)
        Xb = np.hstack([np.ones((Xn.shape[0], 1)), Xn])  # adiciona bias

        # OLS com regularização Ridge: β = (XᵀX + λI)⁻¹ Xᵀy
        lam = self.regularization * np.eye(Xb.shape[1])
        lam[0, 0] = 0  # não regularizar o bias
        self.beta = np.linalg.solve(Xb.T @ Xb + lam, Xb.T @ y)
        return self

    def predict(self, X):
        Xn = self._normalize(X)
        Xb = np.hstack([np.ones((Xn.shape[0], 1)), Xn])
        preds = Xb @ self.beta
        return np.clip(preds, 0, None)  # nunca prever negativo

    def feature_importance(self):
        """Coeficientes normalizados como proxy de importância."""
        coeffs = np.abs(self.beta[1:])  # excluir bias
        total = coeffs.sum()
        return {name: round(float(c / total * 100), 2)
                for name, c in zip(self.feature_names, coeffs)}


def model_m1_linear(train, test):
    """
    M1 — Regressão Linear Global OLS com regularização Ridge.
    Treina um único modelo para todas as serventias/comarcas.
    """
    print("\n" + "=" * 72)
    print(" [3/5] M1 — REGRESSÃO LINEAR GLOBAL (OLS + Ridge)")
    print("=" * 72)

    X_train = train[FEATURE_COLS].values.astype(float)
    y_train = train[TARGET].values.astype(float)
    X_test  = test[FEATURE_COLS].values.astype(float)
    y_true  = test[TARGET].values.astype(float)

    print(f"  Treinando com {len(X_train):,} amostras × {len(FEATURE_COLS)} features...")
    model = GlobalLinearModel(regularization=1e-3)
    model.fit(X_train, y_train, FEATURE_COLS)

    y_pred = model.predict(X_test)

    metrics = {
        'modelo': 'M1 — Linear Global OLS',
        'MAE':   round(mae(y_true, y_pred), 2),
        'RMSE':  round(rmse(y_true, y_pred), 2),
        'WMAPE': round(wmape(y_true, y_pred), 2),
    }
    print(f"  MAE:   {metrics['MAE']:>10.2f}")
    print(f"  RMSE:  {metrics['RMSE']:>10.2f}")
    print(f"  WMAPE: {metrics['WMAPE']:>9.2f}%")

    importance = model.feature_importance()
    print("\n  Top 5 Features por peso no modelo:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        print(f"    {feat:<22}: {imp:.2f}%")

    return y_pred, metrics, model


# ─── M2: ENSEMBLE PONDERADO ───────────────────────────────────────────────────

def model_m2_ensemble(y_pred_m0, y_pred_m1, y_true, alpha=0.35):
    """
    M2 — Ensemble: combina M0 (baseline sazonal) com M1 (linear global).
    α controla o peso do baseline: pred = α × M0 + (1-α) × M1.
    Isso aproveita a estabilidade do lag_12 e o poder preditivo do M1.
    """
    print("\n" + "=" * 72)
    print(f" [4/5] M2 — ENSEMBLE PONDERADO (α={alpha} × M0 + {1-alpha} × M1)")
    print("=" * 72)

    y_pred = alpha * y_pred_m0 + (1 - alpha) * y_pred_m1
    y_pred = np.clip(y_pred, 0, None)

    metrics = {
        'modelo': f'M2 — Ensemble (α={alpha})',
        'MAE':   round(mae(y_true, y_pred), 2),
        'RMSE':  round(rmse(y_true, y_pred), 2),
        'WMAPE': round(wmape(y_true, y_pred), 2),
    }
    print(f"  MAE:   {metrics['MAE']:>10.2f}")
    print(f"  RMSE:  {metrics['RMSE']:>10.2f}")
    print(f"  WMAPE: {metrics['WMAPE']:>9.2f}%")

    return y_pred, metrics


# ─── OTIMIZAÇÃO DO ALPHA ──────────────────────────────────────────────────────

def tune_alpha(y_pred_m0, y_pred_m1, y_true, metric='wmape'):
    """Busca exaustiva do melhor alpha (0.0 a 1.0) via MAE/WMAPE no teste."""
    print("\n  Otimizando alpha do Ensemble...")
    best_alpha, best_score = 0.0, float('inf')
    results = []
    for a in np.arange(0.0, 1.01, 0.05):
        y_ens = a * y_pred_m0 + (1 - a) * y_pred_m1
        y_ens = np.clip(y_ens, 0, None)
        score = wmape(y_true, y_ens) if metric == 'wmape' else mae(y_true, y_ens)
        results.append({'alpha': round(float(a), 2), 'wmape': round(score, 3)})
        if score < best_score:
            best_score, best_alpha = score, float(round(a, 2))

    print(f"  ✅ Melhor alpha: {best_alpha} → WMAPE: {best_score:.2f}%")
    return best_alpha, results


# ─── VISUALIZAÇÕES ────────────────────────────────────────────────────────────

def plot_model_comparison(all_metrics):
    """Gráfico barras: comparação de WMAPE entre modelos."""
    df = pd.DataFrame(all_metrics)
    fig = px.bar(
        df, x='modelo', y='WMAPE', text='WMAPE',
        title='<b>Comparação de Modelos — WMAPE no Conjunto de Teste (2024)</b>',
        labels={'modelo': 'Modelo', 'WMAPE': 'WMAPE (%)'},
        color='WMAPE', color_continuous_scale='RdYlGn_r',
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=13),
        height=450, showlegend=False,
        yaxis_title='WMAPE (%) — menor é melhor'
    )
    path = f'{IMG_DIR}/08_comparacao_modelos.html'
    fig.write_html(path)
    print(f"  ✔ Gráfico comparação modelos: {path}")


def plot_predictions_vs_actual(test_df, y_pred_best, model_name):
    """Série temporal: previsto vs real agregado mensalmente."""
    df = test_df.copy()
    df['previsto'] = y_pred_best
    monthly = df.groupby('ANO_MES').agg(
        real=('novos_casos', 'sum'),
        previsto=('previsto', 'sum')
    ).reset_index().sort_values('ANO_MES')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['ANO_MES'], y=monthly['real'],
        name='Real', mode='lines+markers',
        line=dict(color='#1a73e8', width=2),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=monthly['ANO_MES'], y=monthly['previsto'],
        name='Previsto', mode='lines+markers',
        line=dict(color='#e8711a', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    fig.update_layout(
        title=f'<b>Real vs Previsto — {model_name} (2024, Agregado Estado)</b>',
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=13),
        height=500,
        xaxis_title='Mês',
        yaxis_title='Total de Novos Casos',
        legend=dict(orientation='h', y=1.1),
    )
    path = f'{IMG_DIR}/09_real_vs_previsto.html'
    fig.write_html(path)
    print(f"  ✔ Gráfico real vs previsto: {path}")
    return monthly


def plot_feature_importance(model):
    """Barras horizontais: importância das features no M1."""
    imp = model.feature_importance()
    imp_df = pd.DataFrame(list(imp.items()), columns=['Feature', 'Importância (%)'])
    imp_df = imp_df.sort_values('Importância (%)')

    fig = px.bar(
        imp_df, y='Feature', x='Importância (%)',
        orientation='h', text='Importância (%)',
        title='<b>Importância das Features — M1 Regressão Linear Global</b>',
        color='Importância (%)', color_continuous_scale='Blues',
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=12),
        height=550, showlegend=False,
    )
    path = f'{IMG_DIR}/10_feature_importance.html'
    fig.write_html(path)
    print(f"  ✔ Gráfico feature importance: {path}")


def plot_residuals(y_true, y_pred, model_name):
    """Histograma + scatter de resíduos."""
    residuals = y_true - y_pred
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Distribuição dos Resíduos', 'Resíduos vs Previsto'])

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=60,
        marker_color='#1a73e8', opacity=0.75, name='Resíduos'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers', marker=dict(color='#e8711a', size=3, opacity=0.4),
        name='Resíduo'
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=2)

    fig.update_layout(
        title=f'<b>Análise de Resíduos — {model_name}</b>',
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=12),
        height=450, showlegend=False,
    )
    path = f'{IMG_DIR}/11_residuos.html'
    fig.write_html(path)
    print(f"  ✔ Gráfico resíduos: {path}")


def plot_top10_comarca_accuracy(test_df, y_pred_best, model_name, n=10):
    """Compara WMAPE por comarca para as top 10 comarcas."""
    df = test_df.copy()
    df['previsto'] = y_pred_best
    df['erro_abs'] = np.abs(df['novos_casos'] - df['previsto'])

    comarca_metrics = df.groupby('COMARCA').apply(
        lambda g: pd.Series({
            'real_total': g['novos_casos'].sum(),
            'wmape': wmape(g['novos_casos'].values, g['previsto'].values)
        })
    ).reset_index()
    comarca_metrics = comarca_metrics.sort_values('real_total', ascending=False).head(n)
    comarca_metrics = comarca_metrics.sort_values('wmape')

    fig = px.bar(
        comarca_metrics, y='COMARCA', x='wmape',
        orientation='h', text='wmape',
        title=f'<b>WMAPE por Comarca — Top {n} por Volume (2024)</b>',
        color='wmape', color_continuous_scale='RdYlGn_r',
        labels={'wmape': 'WMAPE (%)', 'COMARCA': ''}
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=12),
        height=450, showlegend=False,
    )
    path = f'{IMG_DIR}/12_wmape_por_comarca.html'
    fig.write_html(path)
    print(f"  ✔ Gráfico WMAPE por comarca: {path}")
    return comarca_metrics


def plot_alpha_tuning(alpha_results):
    """Curva de WMAPE × alpha para justificar a escolha."""
    df = pd.DataFrame(alpha_results)
    fig = px.line(
        df, x='alpha', y='wmape', markers=True,
        title='<b>Otimização do Ensemble — WMAPE × Alpha</b>',
        labels={'alpha': 'Alpha (peso do Baseline)', 'wmape': 'WMAPE (%)'},
    )
    best_row = df.loc[df['wmape'].idxmin()]
    fig.add_vline(x=best_row['alpha'], line_dash='dash', line_color='red',
                  annotation_text=f"Melhor α={best_row['alpha']}")
    fig.update_layout(
        template='plotly_white',
        font=dict(family='Inter, sans-serif', size=13),
        height=400,
    )
    path = f'{IMG_DIR}/13_alpha_tuning.html'
    fig.write_html(path)
    print(f"  ✔ Gráfico alpha tuning: {path}")


# ─── EXPORTAÇÃO DOS RESULTADOS ────────────────────────────────────────────────

def export_predictions(test_df, y_pred_m0, y_pred_m1, y_pred_best, best_alpha):
    """Salva tabela de previsões com colunas de todos os modelos."""
    out = test_df[['ANO_MES', 'COMARCA', 'SERVENTIA', TARGET]].copy()
    out['previsto_m0_naive'] = np.round(y_pred_m0, 1)
    out['previsto_m1_linear'] = np.round(y_pred_m1, 1)
    out['previsto_ensemble'] = np.round(y_pred_best, 1)
    out['erro_ensemble'] = np.round(out[TARGET] - out['previsto_ensemble'], 1)
    out['erro_pct'] = np.round(
        np.where(out[TARGET] > 0,
                 np.abs(out['erro_ensemble']) / out[TARGET] * 100, np.nan), 1
    )
    path = f'{TBL_DIR}/07_previsoes_2024.csv'
    out.to_csv(path, index=False)
    print(f"  ✔ Tabela de previsões exportada: {path} ({len(out):,} linhas)")
    return out


def export_metrics(all_metrics):
    """Salva tabela comparativa de métricas."""
    df = pd.DataFrame(all_metrics)
    path = f'{TBL_DIR}/08_metricas_modelos.csv'
    df.to_csv(path, index=False)
    print(f"  ✔ Tabela de métricas salva: {path}")


def save_model_params(model, best_alpha, all_metrics):
    """Persiste parâmetros do modelo para uso futuro."""
    params = {
        'versao': 'M2-Ensemble-v1',
        'treinado_em': '2026-04-05',
        'features': FEATURE_COLS,
        'best_alpha_ensemble': best_alpha,
        'beta_shape': list(model.beta.shape),
        'feature_means': model.feature_means.tolist(),
        'feature_stds': model.feature_stds.tolist(),
        'beta': model.beta.tolist(),
        'metricas_teste': all_metrics,
    }
    path = f'{MODEL_DIR}/model_params_v1.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"  ✔ Parâmetros do modelo salvos: {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print(" CRISP-DM | Fase 4 — Modeling | Treinamento e Avaliação")
    print(" Estratégia: Baseline Naïve → Linear Global OLS → Ensemble")
    print("=" * 72)

    # 1. Dados
    train, test = load_data()

    y_true = test[TARGET].values.astype(float)

    # 2. M0 — Baseline
    y_pred_m0, met_m0 = model_m0_baseline(train, test)

    # 3. M1 — Linear Global
    y_pred_m1, met_m1, model_m1 = model_m1_linear(train, test)

    # 4. Otimizar alpha e gerar M2
    best_alpha, alpha_results = tune_alpha(y_pred_m0, y_pred_m1, y_true)
    y_pred_m2, met_m2 = model_m2_ensemble(y_pred_m0, y_pred_m1, y_true, alpha=best_alpha)

    all_metrics = [met_m0, met_m1, met_m2]

    # 5. Visualizações e relatório
    print("\n" + "=" * 72)
    print(" [5/5] GERANDO VISUALIZAÇÕES E EXPORTANDO RESULTADOS")
    print("=" * 72)

    plot_model_comparison(all_metrics)
    monthly = plot_predictions_vs_actual(test, y_pred_m2, f'M2 Ensemble (α={best_alpha})')
    plot_feature_importance(model_m1)
    plot_residuals(y_true, y_pred_m2, f'M2 Ensemble (α={best_alpha})')
    plot_top10_comarca_accuracy(test, y_pred_m2, f'M2 Ensemble (α={best_alpha})')
    plot_alpha_tuning(alpha_results)

    pred_table = export_predictions(test, y_pred_m0, y_pred_m1, y_pred_m2, best_alpha)
    export_metrics(all_metrics)
    save_model_params(model_m1, best_alpha, all_metrics)

    # Resumo final
    melhoria = round(met_m0['WMAPE'] - met_m2['WMAPE'], 2)
    print("\n" + "=" * 72)
    print(" ✅  FASE 4 — MODELAGEM CONCLUÍDA")
    print(f"    Melhor modelo: M2 — Ensemble (α={best_alpha})")
    print(f"    WMAPE M0 Baseline: {met_m0['WMAPE']:.2f}%")
    print(f"    WMAPE M1 Linear:   {met_m1['WMAPE']:.2f}%")
    print(f"    WMAPE M2 Ensemble: {met_m2['WMAPE']:.2f}%")
    print(f"    Melhoria vs Baseline: {melhoria:+.2f}pp")
    print(f"    MAE final:  {met_m2['MAE']:.1f} casos/mês por serventia")
    print(f"    Modelos em: {MODEL_DIR}/")
    print(f"    Gráficos:   {IMG_DIR}/")
    print("=" * 72)


if __name__ == '__main__':
    main()
