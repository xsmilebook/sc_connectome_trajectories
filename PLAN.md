# Plan

Stabilize CLG-ODE training by normalizing topo features with a data-driven log compression, reweighting only topo/manifold via GradNorm, and adding a 20% cosine warmup for topo, while keeping the existing vel/acc masking + schedule intact.

## Scope
- In: Analyze current loss imbalance, implement `topo_norm = log1p(raw_topo / scale)` using q80/q90 scale, add GradNorm for `{topo, manifold}` only, add 20% cosine warmup on topo weight, expose new flags, and document the updated training strategy.
- Out: Data regeneration, edits under `data/` or `outputs/`, structural refactors, or changes to vel/acc schedules beyond existing masks.

## Action items
[ ] Review `outputs/results/clg_ode/.../fold0/metrics.csv` and training logs to quantify topo vs manifold magnitudes and confirm where normalization should hook in.
[ ] Implement data-driven topo scaling (q80/q90 of raw topo bins) and `log1p` compression in the topo pipeline, deciding whether to apply before or after z-score.
[ ] Add GradNorm weighting for `{topo, manifold}` only, leaving vel/acc governed by existing masks/schedules.
[ ] Add a 20% cosine warmup for topo loss weight (or GradNorm target) to avoid early domination.
[ ] Add CLI/config flags for `topo_scale_quantile`, `topo_log_compress`, `gradnorm_scope`, and `topo_warmup_frac`, and wire defaults into `scripts/train_clg_ode.py`.
[ ] Update `docs/methods.md` and `docs/workflow.md` (if usage changes) plus `PROGRESS.md` and `docs/sessions/` to record the strategy.
[ ] Run a quick validation (one fold, short epochs) to ensure loss scales and training stability improve without breaking metrics logging.
[ ] Commit the change set with a clear message.

## Open questions
- Default scale quantile: choose q0.8 vs q0.9, or expose both via flag with q0.9 default?
