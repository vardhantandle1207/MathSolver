#!/usr/bin/env bash
# Set this project up on a shared GPU box, entirely in $HOME.
#
# Written for a machine with no sudo, no conda, and a system Python too old for
# the codebase (3.8 can't parse `str | None`). Everything lands under ~/apps and
# ~/mathsolver; nothing is installed system-wide and no existing file is touched.
#
#   bash scripts/setup_gpu_server.sh            # uses GPU 0
#   GPU=2 bash scripts/setup_gpu_server.sh      # pick a different card
#
# Re-running is safe: each step is skipped if it's already done.
set -euo pipefail

GPU="${GPU:-0}"
MODEL="${MODEL:-qwen2.5:14b-instruct}"
APPS="$HOME/apps"
PROJECT="${PROJECT:-$HOME/mathsolver}"
OLLAMA_DIR="$APPS/ollama"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama-models}"

echo "=== 1/5  Ollama (user-local, no sudo) ==="
# Ollama ships .tar.zst now, and a shared box typically has no zstd and no way
# to apt-get one. Python's zstandard wheel needs no admin rights, so the archive
# is decompressed with that and handed to tar as a plain stream.
if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
  mkdir -p "$OLLAMA_DIR"
  TAG=$(curl -s https://api.github.com/repos/ollama/ollama/releases/latest \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
  URL="https://github.com/ollama/ollama/releases/download/$TAG/ollama-linux-amd64.tar.zst"
  echo "downloading ollama $TAG (~1.4GB)..."
  curl -fL --progress-bar "$URL" -o "$APPS/ollama.tar.zst"

  if command -v zstd >/dev/null; then
    zstd -dc "$APPS/ollama.tar.zst" | tar -x -C "$OLLAMA_DIR"
  else
    python3 -m pip install --user --quiet zstandard
    python3 - "$APPS/ollama.tar.zst" "$OLLAMA_DIR" <<'PY'
import subprocess, sys, zstandard
archive, dest = sys.argv[1], sys.argv[2]
tar = subprocess.Popen(["tar", "-x", "-C", dest], stdin=subprocess.PIPE)
with open(archive, "rb") as fh:
    zstandard.ZstdDecompressor().copy_stream(fh, tar.stdin)
tar.stdin.close()
raise SystemExit(tar.wait())
PY
  fi
  rm -f "$APPS/ollama.tar.zst"
else
  echo "already installed"
fi
export PATH="$OLLAMA_DIR/bin:$PATH"
ollama --version || true

echo
echo "=== 2/5  Start the Ollama server on GPU $GPU ==="
if curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "a server is already listening on 11434 — reusing it"
else
  mkdir -p "$OLLAMA_MODELS"
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$OLLAMA_DIR/bin/ollama" serve \
    > "$HOME/ollama-serve.log" 2>&1 &
  echo "waiting for it to come up..."
  for _ in $(seq 1 30); do
    curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
  curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null \
    && echo "server up (log: ~/ollama-serve.log)" \
    || { echo "server failed to start — see ~/ollama-serve.log"; exit 1; }
fi

echo
echo "=== 3/5  Python 3.12 via uv (system python is 3.8, too old) ==="
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo
echo "=== 4/5  Project + dependencies ==="
if [ ! -d "$PROJECT/.git" ]; then
  git clone https://github.com/vardhantandle1207/MathSolver.git "$PROJECT"
else
  echo "repo present; pulling latest"
  git -C "$PROJECT" pull --ff-only || true
fi
cd "$PROJECT"
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python -r requirements.txt

cat > .env <<EOF
LLM_PROVIDER=ollama
OLLAMA_MODEL=$MODEL
OLLAMA_HOST=http://localhost:11434
MAX_STEPS=10
TEMPERATURE=0.0
EXEC_TIMEOUT=10
MAX_RETRIES=3
MAX_REPEATS=3
MAX_NUDGES=2
MAX_TOKENS_BUDGET=30000
MAX_PROBLEM_CHARS=4000
EOF
echo "wrote .env (model: $MODEL)"

echo
echo "=== 5/5  Pull the model and smoke-test ==="
ollama pull "$MODEL"
.venv/bin/python -m pytest -q
.venv/bin/python -c "
from src.agent import solve
r = solve('Evaluate the definite integral of x^2 from 0 to 3.')
print('smoke:', repr(r.answer), '| stop:', r.stopped_reason, '| steps:', r.steps_taken)
"

echo
echo "=== READY ==="
echo "project: $PROJECT"
echo "every new shell needs:"
echo "  export PATH=\"$OLLAMA_DIR/bin:\$HOME/.local/bin:\$PATH\""
echo
echo "run the benchmarks (nohup so they survive a dropped SSH session):"
echo "  cd $PROJECT"
echo "  nohup .venv/bin/python -u -m src.evaluate --data data/jeemains_math.json \\"
echo "      --workers 4 --out results/jeemains_14b.json > results/jeemains_14b.log 2>&1 &"
echo "  tail -f results/jeemains_14b.log"
