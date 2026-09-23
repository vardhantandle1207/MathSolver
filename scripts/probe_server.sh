#!/usr/bin/env bash
# One-shot survey of a GPU box: what it has, what it can reach, what's already
# installed. Read-only — installs nothing, changes nothing.
echo "=== HOST ==="; hostname; uname -sr; echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv 2>/dev/null \
  || echo "nvidia-smi not found"
echo
echo "=== CPU / RAM / DISK ==="
nproc 2>/dev/null | sed 's/^/cores: /'
free -g 2>/dev/null | awk '/Mem:/{print "RAM: "$2" GB total, "$7" GB available"}'
df -h "$HOME" | tail -1 | awk '{print "home disk: "$4" free of "$2}'
echo
echo "=== PYTHON ==="
for p in python3 python; do command -v $p >/dev/null && echo "$p -> $($p -V 2>&1)"; done
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null \
  || echo "torch: not installed"
echo
echo "=== TOOLING ==="
for c in ollama git pip3 conda module srun docker; do
  command -v $c >/dev/null && echo "$c: yes ($(command -v $c))" || echo "$c: no"
done
echo
echo "=== NETWORK (5s timeouts) ==="
for url in https://huggingface.co https://ollama.com https://pypi.org; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null)
  echo "$url -> ${code:-unreachable}"
done
echo "proxy vars: ${http_proxy:-none} ${https_proxy:-none}"
echo
echo "=== PERMISSIONS ==="
touch "$HOME/.probe_write_test" 2>/dev/null && echo "home writable: yes" && rm -f "$HOME/.probe_write_test" || echo "home writable: NO"
echo "sudo: $(sudo -n true 2>/dev/null && echo yes || echo 'no (or needs password)')"
