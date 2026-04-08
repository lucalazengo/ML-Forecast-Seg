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
