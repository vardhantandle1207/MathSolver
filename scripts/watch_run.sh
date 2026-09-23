#!/usr/bin/env bash
# Mirror the GPU-server run onto this machine and print a live summary.
# The work happens remotely; this just makes it visible locally.
#   bash scripts/watch_run.sh
SERVER="${SERVER:-ai25mtech11004@192.168.209.81}"
REMOTE_LOG="${REMOTE_LOG:-~/mathsolver/results/jeemains_14b.log}"
LOCAL_LOG="${LOCAL_LOG:-results/jeemains_14b_live.log}"
TOTAL="${TOTAL:-475}"

while true; do
  scp -q "$SERVER:$REMOTE_LOG" "$LOCAL_LOG" 2>/dev/null
  clear
  echo "=== JEE Main $TOTAL · qwen2.5:14b · $(date +%H:%M:%S) ==="
  awk -v total="$TOTAL" '/^\[/{n++; if($0~/base:OK/)b++; if($0~/agent:OK/)a++}
    END{ if(n) printf "done %d/%d (%.1f%%)\n\nbaseline %d/%d = %.1f%%\nagent    %d/%d = %.1f%%\n",
          n,total,100*n/total, b,n,100*b/n, a,n,100*a/n; else print "waiting for first result..." }' "$LOCAL_LOG"
  echo
  echo "--- latest ---"
  tail -5 "$LOCAL_LOG" 2>/dev/null
  sleep 30
done
