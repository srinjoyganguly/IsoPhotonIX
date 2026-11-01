#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f frontend/dist/index.html ]; then
  echo "Build the interface first: cd frontend && pnpm install --frozen-lockfile && pnpm build" >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT:-8000}"
