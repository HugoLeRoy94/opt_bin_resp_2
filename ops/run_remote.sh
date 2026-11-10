#!/usr/bin/env bash
# Launch a task on the cluster in a detached tmux session, with a GPU memory
# monitor alongside it for the whole run.
#
# Usage:
#   ops/run_remote.sh <task> <script> [gpu] [-- script args...]
#
# Examples:
#   ops/run_remote.sh analytic_baseline train_sweep_R
#   ops/run_remote.sh analytic_baseline train_sweep_R 2 -- --R 2 4 8
#   ops/run_remote.sh fig2 train_complete_heteromers 1
#   ops/run_remote.sh fig2 test_counting 0
#
# <task> is a directory under tasks/ and <script> a module in it, without .py.
# Scripts run as `python -m tasks.<task>.<script>` from /app, which is what puts
# the packages on the path; running the file by its path would not.
#
# The run lives in a detached tmux session, so it survives logout.
#   attach:  ssh -t <server> tmux attach -t <session>
#   list:    ssh -t <server> tmux ls
#   kill:    ssh -t <server> tmux kill-session -t <session>
# NEVER `tmux kill-server`: that kills every session on a shared machine.
set -e

SERVER="${OCTOPUS_DATA_SERVER:-leroy@10.187.172.7}"
REMOTE_ROOT="${OCTOPUS_REMOTE_ROOT:-/home/leroy/opt_bin_resp_2}"
REMOTE_DATA="${OCTOPUS_REMOTE_DATA_ROOT:-/storage/leroy/data}"
COMPOSE="$REMOTE_ROOT/docker-compose.server.yaml"

TASK="$1"; SCRIPT="$2"; GPU="${3:-0}"
shift $(( $# < 3 ? $# : 3 ))
[ "${1:-}" = "--" ] && shift
ARGS="$*"

if [ -z "${TASK:-}" ] || [ -z "${SCRIPT:-}" ]; then
    echo "usage: $0 <task> <script> [gpu] [-- args]"; exit 1
fi
SCRIPT="${SCRIPT%.py}"

SESSION="${TASK}_${SCRIPT}_$(date +%H%M%S)"
# Every task writes to <data root>/<task>, so the logs sit with the runs and
# ops/manage_data.py sync pulls them back together.
MEM_LOG="$REMOTE_DATA/${TASK}/gpu_mem_${SESSION}.csv"
RUN_LOG="$REMOTE_DATA/${TASK}/run_${SESSION}.log"

# Pull first, visibly: fail before anything is launched rather than halfway in.
ssh "$SERVER" "cd $REMOTE_ROOT && git pull"

MON="mkdir -p $REMOTE_DATA/$TASK; $REMOTE_ROOT/ops/monitor_gpu.sh $GPU 5 $MEM_LOG >/dev/null 2>&1 &"
# -T disables the container TTY so output can be piped to tee. python3 -u:
# without a TTY Python block-buffers stdout, so prints would not reach the pane
# or the log until the buffer filled.
SIM="MY_GPU=$GPU docker compose -f $COMPOSE run --rm -T gpu-runner \
python3 -u -m tasks.${TASK}.${SCRIPT} --data_root /app/data/${TASK} $ARGS 2>&1 | tee -a $RUN_LOG"
# On exit, stop the monitor by its recorded PID and move the logs into the sweep.
# No pattern-based pkill: the pattern would also match this session's own shell,
# killing the pane and discarding a crash message. `exec bash` keeps it open.
FIN="$REMOTE_ROOT/ops/finalize_run.sh '$RUN_LOG' '$MEM_LOG' '$REMOTE_DATA'; \
echo; echo '=== run finished (Ctrl-b d to detach) ==='; exec bash"

ssh "$SERVER" tmux new-session -d -s "$SESSION" "bash -lc \"$MON $SIM; $FIN\""

echo "launched tmux session: $SESSION"
echo "  attach:   ssh -t $SERVER tmux attach -t $SESSION"
echo "  list:     ssh -t $SERVER tmux ls"
echo "  kill:     ssh -t $SERVER tmux kill-session -t $SESSION"
echo "  run log:  $SERVER:$RUN_LOG"
echo "  gpu log:  $SERVER:$MEM_LOG"
