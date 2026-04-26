Report the current state of the Neuroprobe cross-subject hillclimb. Tight, factual, no headers.

## Steps

1. **Read** `docs/neuroprobe/plan.md` (first 80 lines is enough) to identify the active hillclimb stage (0 reproduce / 1 cold-start / 2 SSL pretrain / 3 submit) and the gate thresholds (abort < 0.539, submit ≥ 0.56, stretch ≥ 0.58).

2. **Tail** `docs/experiments/v14_ablation_log.csv` — last ~30 rows. Filter for Neuroprobe cross-subject runs (look for `neuroprobe` / `cross_subject` in `experiment_id` or comparable column). Pull the best score landed so far. If no Neuroprobe rows yet, say "no Neuroprobe results in CSV yet."

3. **Tail** `.ablation_submissions.jsonl` — last 10 entries. Cross-reference with `ssh ht203@dcc-login.oit.duke.edu squeue -u ht203` (only if user hasn't blocked DCC access this session). Report Neuroprobe-tagged jobs that are still PENDING / RUNNING.

4. **Report** in 5–8 lines:
   - Stage: N (one phrase, e.g. "Stage 1 cold-start, target ≥ 0.539")
   - Best score: X — distance to gates (abort / submit / stretch), verdict (on track / at risk / submit-ready / abort)
   - In-flight: `<job_id> <name>` × N, or "none"
   - Last submitted: `<job_id> <name>` (`ts` as ISO date)
   - Next: the next action from `docs/neuroprobe/plan.md`

Skip prose. If you can't read a file (CSV column drift, missing plan section), say so on the line where it would have gone — don't fail the whole report.
