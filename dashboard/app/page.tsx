"use client";

import { useState, useEffect, useMemo } from 'react';
import ForecastChart from '../components/ForecastChart';

const MONTH_NAMES = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'];

// Anos disponíveis com labels descritivos
const YEAR_OPTIONS: { value: string; label: string }[] = [
  { value: 'Tudo', label: 'Todos os Anos' },
  { value: '2020', label: '2020' },
  { value: '2021', label: '2021' },
  { value: '2022', label: '2022' },
  { value: '2023', label: '2023' },
  { value: '2024', label: '2024' },
  { value: '2025', label: '2025 (Teste)' },
  { value: '2026', label: '2026 (Previsão)' },
];

export default function DashboardClient() {
  const [comarca, setComarca] = useState<string>("Tudo");
  const [serventia, setServentia] = useState<string>("Tudo");
  const [ano, setAno] = useState<string>("2026");
  const [mes, setMes] = useState<string>("Tudo");
  const [loading, setLoading] = useState<boolean>(true);

  const [chartData, setChartData] = useState<any[]>([]);
  const [fullData, setFullData] = useState<any>(null);
  const [hierarquia, setHierarquia] = useState<any>({});
  const [kpis, setKpis] = useState<any>(null);

  // Fetch real data on mount
  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch('/forecast_data_1.json').then(r => r.json()).catch(() => null),
      fetch('/forecast_data_2.json').then(r => r.json()).catch(() => null),
      fetch('/hierarquia.json').then(r => r.json()).catch(() => null),
      fetch('/kpis.json').then(r => r.json()).catch(() => null)
    ]).then(([d1, d2, h, k]) => {
      if (d1 || d2) setFullData({ ...d1, ...d2 });
      if (h) setHierarquia(h);
      if (k) setKpis(k);
      setLoading(false);
    });
  }, []);

  // Reset serventia when comarca changes
  useEffect(() => {
    if (comarca !== "Tudo" && serventia !== "Tudo") {
      const validServentias = hierarquia[comarca] || [];
      if (!validServentias.includes(serventia)) {
        setServentia("Tudo");
      }
    }
  }, [comarca, hierarquia, serventia]);

  // Update chart data whenever filters or fullData changes
  useEffect(() => {
    if (!fullData) return;

    let data: any[] = [];
    const node = fullData[comarca]?.[serventia] || {};

    if (ano === "Tudo") {
      const anos = Object.keys(node).sort();
      data = anos.map(a => {
        let sumHist: number | null = null;
        let sumPrev: number | null = null;
        let sumPMin: number | null = null;
        let sumPMax: number | null = null;

        for (const m of MONTH_NAMES) {
          const val = node[a]?.[m];
          if (val) {
            if (val.historico !== null) sumHist = (sumHist || 0) + val.historico;
            if (val.previsao !== null) sumPrev = (sumPrev || 0) + val.previsao;
            if (val.previsao_min !== null) sumPMin = (sumPMin || 0) + val.previsao_min;
            if (val.previsao_max !== null) sumPMax = (sumPMax || 0) + val.previsao_max;
          }
        }
        return {
          mes: a,
          historico: sumHist,
          previsao: sumPrev,
          errorBand: sumPrev !== null ? [sumPMin, sumPMax] : null
        };
      });
    } else {
      const nodeYear = node[ano] || {};
      if (mes === "Tudo") {
        data = MONTH_NAMES.map(m => {
          const val = nodeYear[m] || { historico: null, previsao: null, previsao_min: null, previsao_max: null };
          return {
            mes: m,
            historico: val.historico,
            previsao: val.previsao,
            errorBand: val.previsao !== null ? [val.previsao_min, val.previsao_max] : null
          };
        });
      } else {
        const val = nodeYear[mes] || { historico: null, previsao: null, previsao_min: null, previsao_max: null };
        data = [{
          mes: mes,
          historico: val.historico,
          previsao: val.previsao,
          errorBand: val.previsao !== null ? [val.previsao_min, val.previsao_max] : null
        }];
      }
    }
    setChartData(data);
  }, [comarca, serventia, ano, mes, fullData]);

  // Compute total cases for active filter
  const totalCasos = useMemo(() => {
    if (!chartData.length) return null;
    let totalHist = 0;
    let totalPrev = 0;
    let hasHist = false;
    let hasPrev = false;

    for (const d of chartData) {
      if (d.historico !== null && d.historico !== undefined) {
        totalHist += d.historico;
        hasHist = true;
      }
      if (d.previsao !== null && d.previsao !== undefined) {
        totalPrev += d.previsao;
        hasPrev = true;
      }
    }
    return {
      historico: hasHist ? totalHist : null,
      previsao: hasPrev ? totalPrev : null
    };
  }, [chartData]);

  // Derived lists for selects
  const comarcasList = ["Tudo", ...Object.keys(hierarquia).sort()];
  const serventiasList = comarca === "Tudo" ? ["Tudo"] : ["Tudo", ...(hierarquia[comarca] || []).sort()];

  const handleComarcaChange = (e: any) => {
    setComarca(e.target.value);
    setServentia("Tudo");
  };

  // Format number with dots as thousands separator
  const fmt = (n: number | null) => n !== null ? n.toLocaleString('pt-BR') : '—';

  // Active filter label
  const filterLabel = () => {
    const parts: string[] = [];
    if (ano !== 'Tudo') parts.push(ano);
    if (mes !== 'Tudo') parts.push(mes);
    if (comarca !== 'Tudo') parts.push(comarca);
    if (serventia !== 'Tudo') parts.push(serventia);
    return parts.length > 0 ? parts.join(' / ') : 'Todos os filtros';
  };

  return (
    <>
      <div className="header animate-fade-in">
        <div>
          <h1 className="page-title">Projeção da Demanda</h1>
          <p className="page-subtitle">Acompanhe a previsão e alocação de recursos da justiça estadual usando o modelo LightGBM.</p>
        </div>
      </div>

      {/* Métricas do modelo em um único card + Total de casos */}
      <div className="animate-fade-in delay-100" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* Card único de métricas do modelo */}
        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '12px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Desempenho do Modelo (Teste 2025)
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
            <div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginBottom: '4px' }}>R²</p>
              <p style={{ fontSize: '1.6rem', fontWeight: 700, margin: 0, color: 'var(--success-color)' }}>{kpis ? kpis.r2 : '...'}</p>
            </div>
            <div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginBottom: '4px' }}>WMAPE</p>
              <p style={{ fontSize: '1.6rem', fontWeight: 700, margin: 0, color: 'var(--success-color)' }}>{kpis ? kpis.wmape : '...'}</p>
            </div>
            <div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginBottom: '4px' }}>RMSE</p>
              <p style={{ fontSize: '1.6rem', fontWeight: 700, margin: 0, color: 'var(--accent-color)' }}>{kpis ? kpis.rmse : '...'}</p>
            </div>
            <div>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginBottom: '4px' }}>MAE</p>
              <p style={{ fontSize: '1.6rem', fontWeight: 700, margin: 0, color: 'var(--accent-color)' }}>{kpis ? kpis.mae : '...'}</p>
            </div>
          </div>
        </div>

        {/* Card de total de casos (dinâmico com filtros) */}
        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '12px', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Total de Casos — {filterLabel()}
          </p>
          <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-end' }}>
            {totalCasos?.historico !== null && (
              <div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginBottom: '4px' }}>Histórico (Real)</p>
                <p style={{ fontSize: '2rem', fontWeight: 700, margin: 0, color: 'var(--success-color)' }}>
                  {fmt(totalCasos?.historico ?? null)}
                </p>
              </div>
            )}
            {totalCasos?.previsao !== null && (
              <div>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', marginBottom: '4px' }}>Previsão (Modelo)</p>
                <p style={{ fontSize: '2rem', fontWeight: 700, margin: 0, color: 'var(--accent-color)' }}>
                  {fmt(totalCasos?.previsao ?? null)}
                </p>
              </div>
            )}
            {totalCasos?.historico === null && totalCasos?.previsao === null && (
              <p style={{ color: 'var(--text-secondary)', fontSize: '1rem' }}>Sem dados para os filtros selecionados</p>
            )}
          </div>
        </div>
      </div>

      <div className="filter-bar animate-fade-in delay-200">
        <div className="filter-group">
          <label className="filter-label">Ano</label>
          <select className="select-input" value={ano} onChange={e => setAno(e.target.value)}>
            {YEAR_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">Mês</label>
          <select className="select-input" value={mes} onChange={e => setMes(e.target.value)}>
            <option value="Tudo">Todos os Meses</option>
            <option value="Jan">Janeiro</option>
            <option value="Fev">Fevereiro</option>
            <option value="Mar">Março</option>
            <option value="Abr">Abril</option>
            <option value="Mai">Maio</option>
            <option value="Jun">Junho</option>
            <option value="Jul">Julho</option>
            <option value="Ago">Agosto</option>
            <option value="Set">Setembro</option>
            <option value="Out">Outubro</option>
            <option value="Nov">Novembro</option>
            <option value="Dez">Dezembro</option>
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">Comarca</label>
          <select className="select-input" value={comarca} onChange={handleComarcaChange}>
            {comarcasList.map(c => (
              <option key={c} value={c}>{c === "Tudo" ? "Todas as Comarcas" : c}</option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">Serventia</label>
          <select className="select-input" value={serventia} onChange={e => setServentia(e.target.value)}>
            {serventiasList.map(s => (
              <option key={s} value={s}>{s === "Tudo" ? "Todas as Serventias" : s}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="animate-fade-in delay-300" style={{ display: 'grid', gridTemplateColumns: '1fr' }}>
        <div className="glass-panel">
          <div style={{ marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 600, color: 'var(--text-primary)' }}>Evolução de Casos</h3>
          </div>

          <div style={{ position: 'relative' }}>
            {loading ? (
              <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(15,17,26,0.5)', zIndex: 5, borderRadius: '8px' }}>
                <div style={{ width: '40px', height: '40px', border: '3px solid rgba(255,255,255,0.1)', borderTop: '3px solid var(--accent-color)', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
              </div>
            ) : null}
            <ForecastChart data={chartData} />
            <style dangerouslySetInnerHTML={{ __html: `
              @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            `}} />
          </div>
        </div>
      </div>
    </>
  );
}
