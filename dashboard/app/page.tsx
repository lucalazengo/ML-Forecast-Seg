"use client";

import { useState, useEffect } from 'react';
import ForecastChart from '../components/ForecastChart';

// Componente para a Página Principal do Dashboard
export default function DashboardClient() {
  const [comarca, setComarca] = useState<string>("Tudo");
  const [serventia, setServentia] = useState<string>("Tudo");
  const [ano, setAno] = useState<string>("Tudo");
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
      if (d1 || d2) setFullData({...d1, ...d2});
      if (h) setHierarquia(h);
      if (k) setKpis(k);
      setLoading(false);
    });
  }, []);

  // Update chart data whenever filters or fullData changes
  useEffect(() => {
    if (!fullData) return;
    
    const monthsNames = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'];
    let data = [];

    const node = fullData[comarca]?.[serventia] || {};

    if (ano === "Tudo") {
       const anos = Object.keys(node).sort();
       data = anos.map(a => {
           let sumHist: number | null = null;
           let sumPrev: number | null = null;
           let sumPMin: number | null = null;
           let sumPMax: number | null = null;
           
           for(const m of monthsNames) {
               const val = node[a][m];
               if(val) {
                  if(val.historico !== null) sumHist = (sumHist || 0) + val.historico;
                  if(val.previsao !== null) sumPrev = (sumPrev || 0) + val.previsao;
                  if(val.previsao_min !== null) sumPMin = (sumPMin || 0) + val.previsao_min;
                  if(val.previsao_max !== null) sumPMax = (sumPMax || 0) + val.previsao_max;
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
           data = monthsNames.map(m => {
               const val = nodeYear[m] || { historico: null, previsao: null, previsao_min: null, previsao_max: null };
               return {
                   mes: `${m} ${ano}`,
                   historico: val.historico,
                   previsao: val.previsao,
                   errorBand: val.previsao !== null ? [val.previsao_min, val.previsao_max] : null
               };
           });
       } else {
           const val = nodeYear[mes] || { historico: null, previsao: null, previsao_min: null, previsao_max: null };
           data = [{
               mes: `${mes} ${ano}`,
               historico: val.historico,
               previsao: val.previsao,
               errorBand: val.previsao !== null ? [val.previsao_min, val.previsao_max] : null
           }];
       }
    }
    setChartData(data);
  }, [comarca, serventia, ano, mes, fullData]);

  // Derived lists for selects
  const comarcasList = ["Tudo", ...Object.keys(hierarquia).sort()];
  const serventiasList = comarca === "Tudo" ? ["Tudo"] : ["Tudo", ...(hierarquia[comarca] || []).sort()];

  const handleComarcaChange = (e: any) => {
    setComarca(e.target.value);
    setServentia("Tudo"); // Reset serventia when comarca changes
  };

  return (
    <>
      <div className="header animate-fade-in">
        <div>
          <h1 className="page-title">Projeção da Demanda</h1>
          <p className="page-subtitle">Acompanhe a previsão e alocação de recursos da justiça estadual usando o modelo LightGBM.</p>
        </div>
      </div>

      <div className="metrics-grid animate-fade-in delay-100">
        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>R² (Coeficiente de Det.)</p>
          <h2 style={{ fontSize: '2.4rem', fontWeight: 700, margin: 0, color: 'var(--success-color)' }}>{kpis ? kpis.r2 : '...'}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px', fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Excelente ajuste aos dados</span>
          </div>
        </div>

        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>RMSE</p>
          <h2 style={{ fontSize: '2.4rem', fontWeight: 700, margin: 0, color: 'var(--accent-color)' }}>{kpis ? kpis.rmse : '...'}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px', fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Desvio quadrático médio</span>
          </div>
        </div>

        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>WMAPE</p>
          <h2 style={{ fontSize: '2.4rem', fontWeight: 700, margin: 0, color: 'var(--accent-color)' }}>{kpis ? kpis.mape : '...'}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px', fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--success-color)' }}>Alta precisão geral</span>
          </div>
        </div>

        <div className="glass-panel">
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '8px' }}>MAE</p>
          <h2 style={{ fontSize: '2.4rem', fontWeight: 700, margin: 0, color: 'var(--text-primary)' }}>{kpis ? kpis.mae : '...'}</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px', fontSize: '0.85rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Margem média de casos errados</span>
          </div>
        </div>
      </div>

      <div className="filter-bar animate-fade-in delay-200">
        <div className="filter-group">
          <label className="filter-label">Ano</label>
          <select className="select-input" value={ano} onChange={e => setAno(e.target.value)}>
            <option value="Tudo">Todos os Anos</option>
            {/* The dataset runs approx 2020 to 2025 based on real data */}
            <option value="2020">2020</option>
            <option value="2021">2021</option>
            <option value="2022">2022</option>
            <option value="2023">2023</option>
            <option value="2024">2024 (Teste)</option>
            <option value="2025">2025 (Futuro)</option>
            <option value="2026">2026 (Previsão)</option>
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
            <style dangerouslySetInnerHTML={{__html: `
              @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            `}} />
          </div>
        </div>
      </div>
    </>
  );
}
