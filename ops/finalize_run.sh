#!/usr/bin/env bash
# Move a run's GPU trace and stdout log INTO the sweep directory it created, so
# they sit with that sweep's outputs rather than loose in the task's data root.
#
# The sweep directory name is chosen inside Python and carries a timestamp, so
# the launcher cannot know it up front. It is recovered from the "Sweep: <path>"
# line Sweep.execute prints, captured in the run log. A script that creates no
# sweep (a test_*.py pass) prints no such line, so its logs are left where they
# are rather than guessed at.
#
#   finalize_run.sh <run_log> <mem_log> <remote_data_root>
set -u
RUN_LOG="$1"; MEM_LOG="$2"; REMOTE_DATA="$3"

# Stop the monitor by the PID it recorded. Done with literal arguments so there
# is no pattern that could match the tmux shell and kill the session.
mon_pid=$(cat "$MEM_LOG.pid" 2>/dev/null || true)
[ -n "$mon_pid" ] && kill "$mon_pid" 2>/dev/null || true
rm -f "$MEM_LOG.pid"

sweep_c=$(grep -m1 '^Sweep: ' "$RUN_LOG" 2>/dev/null | awk '{print $NF}')
[ -z "$sweep_c" ] && exit 0                    # no sweep created; leave them

# The container sees /app/data; the host sees the real data root.
sweep_h=${sweep_c/\/app\/data/$REMOTE_DATA}
[ -d "$sweep_h" ] || exit 0

mv -f "$MEM_LOG" "$sweep_h"/ 2>/dev/null || true
mv -f "$RUN_LOG" "$sweep_h"/ 2>/dev/null || true
echo "moved run and gpu logs into $sweep_h"
