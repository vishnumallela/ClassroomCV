# api-service image. The build context is the REPO ROOT, not this directory:
# Bun workspaces resolve `--frozen-lockfile` against the root `bun.lock` plus
# every workspace manifest, so a context rooted here could not install.
# Railway points at this file via `build.dockerfilePath` in .railway/railway.ts.

FROM oven/bun:1.3.14-slim AS deps
WORKDIR /srv
# Manifests only, so this layer re-resolves solely when a dependency changes.
COPY package.json bun.lock bunfig.toml ./
COPY apps/api-service/package.json apps/api-service/
COPY apps/frontend/package.json apps/frontend/
COPY packages/api-contracts/package.json packages/api-contracts/
# --production: the runtime needs drizzle-orm/hono/bullmq, never drizzle-kit,
# oxlint or pino-pretty (NODE_ENV=production picks the plain pino transport).
RUN bun install --frozen-lockfile --production

FROM oven/bun:1.3.14-slim AS runtime
# The analysis worker shells out to ffprobe (duration / fps / dimensions) and
# ffmpeg (thumbnail extraction) — see apps/api-service/src/lib/media.ts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ffprobe -version | head -1

WORKDIR /srv
# The whole tree, not just /srv/node_modules: Bun hoists most packages to the
# root but keeps some in each workspace's own node_modules, and a root-only
# copy loses those (drizzle-orm resolved to nothing at runtime).
COPY --from=deps /srv /srv
COPY package.json bun.lock bunfig.toml tsconfig.base.json ./
COPY packages ./packages
COPY apps/api-service ./apps/api-service

# `db:migrate` reads `./drizzle` relative to the process cwd and `bun run
# start` is a workspace script, so the workspace directory is the workdir.
WORKDIR /srv/apps/api-service

ENV NODE_ENV=production
EXPOSE 8787
CMD ["bun", "run", "start"]
