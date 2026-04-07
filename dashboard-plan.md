# Dashboard Interativo - TJGO Forecast

## Goal
Construir um dashboard interativo (Next.js) na pasta `/dashboard` do repositório atual, que consuma arquivos estáticos (CSV/JSON) gerados pelo modelo LightGBM para visualizar a evolução de casos segmentada por comarca e serventia, auxiliando na alocação de recursos.

## Tasks
- [ ] Tarefa 1: Inicializar projeto Next.js sem Tailwind (Vanilla CSS) na pasta `/dashboard` → Verify: Pasta `/dashboard` criada com `package.json`.
- [ ] Tarefa 2: Criar Design System (Vanilla CSS + CSS Modules) com tema Dark, cores premium e glassmorphism → Verify: `globals.css` configurado e funcionando.
- [ ] Tarefa 3: Desenvolver Componente de Filtros (Comarca, Serventia, Período) → Verify: Componente renderiza no navegador com mock options.
- [ ] Tarefa 4: Integrar biblioteca de gráficos (Ex: Recharts ou Chart.js) e criar os gráficos de evolução temporal → Verify: Gráficos de linha/barras renderizados com dados estáticos.
- [ ] Tarefa 5: Criar serviço de leitura dos arquivos estáticos (JSON/CSV) gerados pelo `train_lgbm.py` → Verify: Dados reais do modelo carregados no console local.
- [ ] Tarefa 6: Montar a página principal (`app/page.tsx`) interligando os filtros e os gráficos → Verify: Filtros alteram a visualização dos gráficos na tela interativamente.
- [ ] Tarefa 7: Otimizar UI/UX (animações de entrada, responsividade, hover effects) → Verify: Interface transmite qualidade premium (WOW factor).

## Done When
- [ ] Next.js app rodando localmente (`npm run dev`).
- [ ] Dashboard exibe os gráficos de evolução corretamente com base nos dados do modelo.
- [ ] Interatividade de filtros funciona, permitindo "drill-down" por segmento.
- [ ] Interface visualmente premium sem uso de placeholders.
