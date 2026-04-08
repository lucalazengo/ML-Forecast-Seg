# Imagem base otimizada para Node.js
FROM node:20-alpine

WORKDIR /app

# Copia primeiro os arquivos de dependência do Dashboard para cache
COPY dashboard/package*.json ./
RUN npm install

# Copia todo o código do dashboard (Isso incluirá a pasta public/ com seus JSONs do modelo)
COPY dashboard/ ./

# Realiza a build da aplicação Next.js
RUN npm run build

# Configura a porta exigida pelo Hugging Face Spaces
ENV PORT=7860
EXPOSE 7860

# Inicia o servidor do Next.js
CMD ["npm", "start"]