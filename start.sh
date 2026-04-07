#!/bin/bash
set -e

echo "🚀 Iniciando processo de deploy e atualização de dados..."

echo "📦 Atualizando as predições a partir dos dados do modelo (via Docker)..."
docker-compose run --rm data-updater

echo "⚙️  Construindo e subindo a imagem do Dashboard..."
docker-compose up -d --build dashboard

echo "✅ Painel atualizado com dados recentes e disponível em: http://localhost:3000"
