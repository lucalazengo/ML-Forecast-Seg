import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Painel Casos Novos | TJGO',
  description: 'Projeção e acompanhamento de casos das Serventias e Comarcas do TJGO.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body>
        <div className="dashboard-container">
          <aside className="sidebar">
            <div className="sidebar-header">
              <div className="logo-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                  <path d="M2 17l10 5 10-5"></path>
                  <path d="M2 12l10 5 10-5"></path>
                </svg>
              </div>
              <div className="logo-text">
                <span className="logo-title">Painel Casos Novos</span>
                <span className="logo-badge">TJGO</span>
              </div>
            </div>

            <div className="sidebar-divider"></div>

            <nav className="nav-links">
              <a href="#" className="nav-link active">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="9"></rect><rect x="14" y="3" width="7" height="5"></rect><rect x="14" y="12" width="7" height="9"></rect><rect x="3" y="16" width="7" height="5"></rect></svg>
                Visão Geral
              </a>
            </nav>

            <div className="sidebar-info-section">
              <p className="sidebar-info-label">Modelo</p>
              <div className="sidebar-info-card">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--success-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                <span>LightGBM (Global)</span>
              </div>

              <p className="sidebar-info-label" style={{marginTop: '1rem'}}>Período de Treino</p>
              <div className="sidebar-info-card">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
                <span>2017 — 2023</span>
              </div>

              <p className="sidebar-info-label" style={{marginTop: '1rem'}}>Validação</p>
              <div className="sidebar-info-card">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                <span>2024 (Out-of-Time)</span>
              </div>
            </div>

            <div className="sidebar-footer">
              <div className="sidebar-divider"></div>
              <div className="sidebar-version">
                <span>ML-Forecast-Seg</span>
                <span className="version-tag">v1.0</span>
              </div>
            </div>
          </aside>
          <main className="main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
