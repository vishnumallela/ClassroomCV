# frontend image: build the Vite SPA with Bun, serve the static bundle with
# Caddy. Built from the REPO ROOT (see apps/api-service/Dockerfile for why).

FROM oven/bun:1.3.14-slim AS build
WORKDIR /srv
COPY package.json bun.lock bunfig.toml ./
COPY apps/api-service/package.json apps/api-service/
COPY apps/frontend/package.json apps/frontend/
COPY packages/api-contracts/package.json packages/api-contracts/
# Not --production here: vite, tailwind and the TanStack router CLI that the
# build runs on are all devDependencies.
RUN bun install --frozen-lockfile

COPY tsconfig.base.json ./
COPY packages ./packages
# @classroom/api-contracts re-exports the api-service router *type*, so the
# api workspace has to be resolvable even though nothing from it is bundled.
COPY apps/api-service ./apps/api-service
COPY apps/frontend ./apps/frontend

# Vite inlines FRONTEND__* at build time, so the API origin is a build ARG,
# not a runtime variable. Railway passes service variables to the builder,
# which is why .railway/railway.ts sets FRONTEND__API_URL on this service.
ARG FRONTEND__API_URL
ENV FRONTEND__API_URL=$FRONTEND__API_URL
RUN cd apps/frontend && bun run build

FROM caddy:2-alpine AS runtime
COPY apps/frontend/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /srv/apps/frontend/dist /srv/dist
