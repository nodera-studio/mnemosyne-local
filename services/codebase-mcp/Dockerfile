# syntax=docker/dockerfile:1
FROM node:22-slim AS build
WORKDIR /app
# tree-sitter 0.25.0 ships NO prebuilt binary — node-gyp compiles it from source, which
# needs a C/C++ toolchain + python3 + the libstdc++ headers. node:22-slim has none, so we
# install them in the BUILD stage only. The compiled .node binary is then carried into the
# slim runtime via the pruned node_modules copy below (so runtime needs no compiler).
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 make g++ \
  && rm -rf /var/lib/apt/lists/*
COPY package.json ./
# Pinned pair: tree-sitter-typescript@0.23.2 declares peer tree-sitter ^0.21.0 but is
# ABI-compatible with 0.25.0 — install with --legacy-peer-deps so npm doesn't reject it.
RUN npm install --legacy-peer-deps
COPY tsconfig.json ./
COPY src ./src
RUN npm run build
# Drop devDependencies but KEEP the compiled native binding under node_modules/tree-sitter*.
RUN npm prune --omit=dev --legacy-peer-deps

FROM node:22-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY package.json ./
# Copy the build stage's pruned (prod-only) node_modules — it carries the compiled
# tree-sitter .node, so the slim runtime loads the native binding WITHOUT a compiler (AC-012).
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY sql ./sql
USER node
EXPOSE 8081
CMD ["node", "dist/server.js"]
