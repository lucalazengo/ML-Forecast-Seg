"use client";

import React from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

interface ForecastChartProps {
  data: any[];
}

const fmt = (n: number | null | undefined) =>
  n != null ? n.toLocaleString('pt-BR') : '—';

export default function ForecastChart({ data }: ForecastChartProps) {
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: 'rgba(15, 17, 26, 0.85)',
          border: '1px solid rgba(255,255,255,0.1)',
          backdropFilter: 'blur(8px)',
          padding: '12px',
          borderRadius: '8px',
          color: '#fff',
          boxShadow: '0 8px 32px 0 rgba(0,0,0,0.4)',
        }}>
          <p style={{ margin: 0, fontWeight: 600, color: '#a0aab2', marginBottom: '8px' }}>{label}</p>
          {payload.map((entry: any, index: number) => {
            if (entry.dataKey === 'errorBand') {
              return (
                <p key={index} style={{ margin: '4px 0', fontSize: '0.85rem', color: 'rgba(255,255,255,0.6)' }}>
                  Margem de Erro: <span style={{ fontWeight: 'bold', color: '#fff' }}>{fmt(entry.value?.[0])} a {fmt(entry.value?.[1])}</span>
                </p>
              );
            }
            return (
              <p key={index} style={{ margin: '4px 0', color: entry.dataKey === 'historico' ? '#10b981' : '#3b82f6' }}>
                {entry.dataKey === 'historico' ? 'Real' : 'Previsão (LightGBM)'}:{' '}
                <span style={{ fontWeight: 'bold', color: '#fff' }}>{fmt(entry.value)} casos</span>
              </p>
            );
          })}
        </div>
      );
    }
    return null;
  };

  const yTickFormatter = (value: number) => {
    if (value >= 1000) return `${(value / 1000).toFixed(0)}k`;
    return String(value);
  };

  return (
    <div style={{ width: '100%', height: 350 }}>
      {data.length === 0 ? (
        <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
          Aguardando seleção de filtros...
        </div>
      ) : (
        <ResponsiveContainer>
          <ComposedChart
            data={data}
            margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="mes" stroke="#a0aab2" tick={{ fill: '#a0aab2' }} axisLine={false} tickLine={false} />
            <YAxis stroke="#a0aab2" tick={{ fill: '#a0aab2' }} axisLine={false} tickLine={false} tickFormatter={yTickFormatter} />
            <Tooltip content={<CustomTooltip />} />

            <Area
              type="monotone"
              dataKey="errorBand"
              fill="#3b82f6"
              fillOpacity={0.15}
              stroke="none"
            />

            <Line
              type="monotone"
              dataKey="historico"
              stroke="#10b981"
              strokeWidth={3}
              dot={{ r: 4, fill: '#0f111a', stroke: '#10b981', strokeWidth: 2 }}
              activeDot={{ r: 6, fill: '#10b981' }}
              connectNulls={false}
            />

            <Line
              type="monotone"
              dataKey="previsao"
              stroke="#3b82f6"
              strokeWidth={3}
              strokeDasharray="5 5"
              dot={{ r: 4, fill: '#0f111a', stroke: '#3b82f6', strokeWidth: 2 }}
              activeDot={{ r: 6, fill: '#3b82f6' }}
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
