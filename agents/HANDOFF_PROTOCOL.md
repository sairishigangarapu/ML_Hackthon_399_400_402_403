# PathSnap Agent Handoff & Continuation Protocol

This directory (`agents/`) maintains durable execution state, checkpoints, and engineering decisions so any agent or new session can seamlessly resume PathSnap development without duplicating completed work or breaking evaluation integrity.

## Files in `agents/`
1. **`STATE.md`** — Human-readable live progress dashboard: current phase, completed stages, active blockers, exact commands to run next.
2. **`CHECKPOINT.json`** — Machine-readable state manifest recording:
   - stage completion status & timestamps
   - random seeds & split guarantees (strict isolation of `train` vs `student_test` vs reserved `evaluator_test`)
   - exact paths of generated datasets, trained model weights, HPO trials, evaluation CSVs, Folium maps, and PDF report
   - verified key metrics from each stage
3. **`DECISIONS.md`** — Chronological architecture & scientific decision record (CRS choice, candidate generation $K$ and radius, monotonic alignment labeling, feature engineering rationale, NN-GLS Vecchia/NNGP formulation, RLCD Brier+Continuity RL formulation, HMM/Viterbi transition math, API/UI security isolation).

## How to Resume in a New Session
1. Read `agents/CHECKPOINT.json` and `agents/STATE.md`.
2. Activate the project virtualenv: `source /mnt/storage/Projects/PESU/ML/Hackathon/PathSnap/.venv/bin/activate`.
3. Identify the first stage in `CHECKPOINT.json` with `"status": "pending"` or `"in_progress"`.
4. Verify upstream artifact paths exist before executing the stage.
5. Upon completing any stage, immediately update BOTH `agents/CHECKPOINT.json` and `agents/STATE.md` (and log any design updates in `agents/DECISIONS.md`).

## Non-Negotiable Integrity Rules
- NEVER use `student_test` for model fitting, scaling, feature selection, or Bayesian hyperparameter search — ONLY for development-stage evaluation & comparison.
- NEVER touch or leak reserved `evaluator_test` labels into the model pipeline or UI.
- NEVER fabricate metrics, plots, or training curves; every number in `report/` must be programmatically read from `results/`.
- NEVER create synthetic epoch loss curves for non-iterative/tree models (LR, RF, LightGBM).
