FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_WORKSPACE=research
ENV VITE_WORKSPACE=$VITE_WORKSPACE
ARG VITE_ANALYSIS_API_URL=/api/v1
ENV VITE_ANALYSIS_API_URL=$VITE_ANALYSIS_API_URL
RUN npm run build

FROM nginx:1.27-alpine
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
