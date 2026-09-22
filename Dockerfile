FROM node:22-alpine AS build
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN npm install --global --no-fund --no-audit pnpm@12.5.1 \
    && pnpm install --frozen-lockfile
COPY . .
ARG VITE_WORKSPACE=research
ENV VITE_WORKSPACE=$VITE_WORKSPACE
ARG VITE_ANALYSIS_API_URL=/api/v1
ENV VITE_ANALYSIS_API_URL=$VITE_ANALYSIS_API_URL
RUN pnpm run build

FROM nginx:1.27-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
