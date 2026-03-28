#!/usr/bin/env bash
# Poll all running experiments every 30 minutes, append to a status log.
set -euo pipefail

TASK_DIR="/private/tmp/claude-501/-Users-bentang-Documents-Code-speech/a2318f71-291c-4395-92ff-f5dd74bb839c/tasks"
LOG="/Users/bentang/Documents/Code/speech/docs/run_status.log"

LOPO="$TASK_DIR/b8i120ci1.output"
SWEEP="$TASK_DIR/b2kl1kgb7.output"
FLAT="$TASK_DIR/brpqjt56d.output"

while true; do
    echo "" >> "$LOG"
    echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG"

    # LOPO
    echo "--- LOPO pilot (8 patients, seed 42) ---" >> "$LOG"
    if [ -f "$LOPO" ]; then
        grep -E "(LOPO target|step [0-9]+:|Stage 2 fold|seed=|PER=)" "$LOPO" 2>/dev/null | tail -10 >> "$LOG"
        # Check if finished
        if grep -q "population_per_mean\|Population" "$LOPO" 2>/dev/null; then
            echo "  STATUS: COMPLETE" >> "$LOG"
        else
            LAST_LINE=$(tail -1 "$LOPO" 2>/dev/null)
            echo "  LATEST: $LAST_LINE" >> "$LOG"
        fi
    else
        echo "  Not started or file missing" >> "$LOG"
    fi

    # Sweep
    echo "--- Sweep (Runs 5-8) ---" >> "$LOG"
    if [ -f "$SWEEP" ]; then
        grep -E "(SWEEP:|seed=)" "$SWEEP" 2>/dev/null | tail -15 >> "$LOG"
        if grep -q "SWEEP COMPLETE" "$SWEEP" 2>/dev/null; then
            echo "  STATUS: COMPLETE" >> "$LOG"
        fi
    fi

    # Flat head
    echo "--- Flat head ablation ---" >> "$LOG"
    if [ -f "$FLAT" ]; then
        grep "seed=" "$FLAT" 2>/dev/null >> "$LOG"
        FLAT_SEEDS=$(grep -c "seed=" "$FLAT" 2>/dev/null || echo 0)
        echo "  Seeds complete: $FLAT_SEEDS/3" >> "$LOG"
    fi

    echo "--- Next check in 30 min ---" >> "$LOG"

    # Exit if all done
    LOPO_DONE=false
    SWEEP_DONE=false
    FLAT_DONE=false
    grep -q "population_per_mean\|Population" "$LOPO" 2>/dev/null && LOPO_DONE=true
    grep -q "SWEEP COMPLETE" "$SWEEP" 2>/dev/null && SWEEP_DONE=true
    [ "$(grep -c 'seed=' "$FLAT" 2>/dev/null || echo 0)" -ge 3 ] && FLAT_DONE=true

    if $LOPO_DONE && $SWEEP_DONE && $FLAT_DONE; then
        echo "" >> "$LOG"
        echo "========== ALL RUNS COMPLETE ==========" >> "$LOG"
        break
    fi

    sleep 1800  # 30 minutes
done
