FROM node:22-alpine AS build
WORKDIR /srv/frontend
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM node:22-alpine
WORKDIR /srv/frontend
COPY --from=build /srv/frontend/dist ./dist
RUN npm install -g serve
EXPOSE 5173
CMD ["serve", "-s", "dist", "-l", "5173"]
