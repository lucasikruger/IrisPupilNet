FROM node:20-bookworm-slim

WORKDIR /app

# Install deps first for layer caching
COPY package.json ./
RUN npm install --no-audit --no-fund

COPY . .

EXPOSE 4322
CMD ["npm", "run", "dev"]
