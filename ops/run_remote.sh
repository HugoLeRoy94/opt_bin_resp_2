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
REMOTE_DATA="${OCTOPUS_REMOTE_DATA_ROOT:-/storage/leroy/data_2}"
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

# ConnectTimeout so an unreachable host fails in seconds rather than hanging.
SSH_OPTS=(-o ConnectTimeout=10)

# Is the repository there at all? `git pull` in a missing directory would just
# fail obscurely, and the first time anyone runs this it will not be there.
if ! ssh "${SSH_OPTS[@]}" "$SERVER" "test -d '$REMOTE_ROOT/.git'" 2>/dev/null; then
    echo "No git repository at $SERVER:$REMOTE_ROOT" >&2
    echo >&2
    echo "Clone it once:" >&2
    echo "  ssh $SERVER 'git clone https://github.com/HugoLeRoy94/opt_bin_resp_2.git $REMOTE_ROOT'" >&2
    echo >&2
    echo "Or point at an existing checkout:" >&2
    echo "  OCTOPUS_REMOTE_ROOT=/path/to/repo $0 $*" >&2
    exit 1
fi

# Pull first, visibly: fail before anything is launched rather than halfway in.
#
# Through `bash -lc`, a LOGIN shell, so the cluster's profile is sourced. A
# plain `ssh host "cmd"` runs a non-login non-interactive shell that reads
# neither .bashrc nor .profile, so http_proxy exported there would be missing
# and the pull would fail to reach github even though an interactive ssh works.
#
# GIT_TERMINAL_PROMPT=0 is what stops this HANGING: ssh allocates no terminal,
# so a credential prompt would go to a pipe nobody reads and the script would
# wait forever with no output. With it, git fails and says so.
if ! ssh "${SSH_OPTS[@]}" "$SERVER" \
        "bash -lc \"cd '$REMOTE_ROOT' && GIT_TERMINAL_PROMPT=0 git pull\""; then
    echo >&2
    echo "git pull failed on the cluster." >&2
    echo >&2
    echo "If it timed out reaching github, the machine is behind the university" >&2
    echo "proxy. Teach git about it once, in git config rather than the shell" >&2
    echo "environment, so it applies to non-login shells like this one:" >&2
    echo "  ssh $SERVER 'git config --global http.proxy  http://proxy.unige.it:8080/'" >&2
    echo "  ssh $SERVER 'git config --global https.proxy http://proxy.unige.it:8080/'" >&2
    echo >&2
    echo "If it asked for credentials, store them once:" >&2
    echo "  ssh -t $SERVER 'cd $REMOTE_ROOT && git config credential.helper store && git pull'" >&2
    exit 1
fi

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
