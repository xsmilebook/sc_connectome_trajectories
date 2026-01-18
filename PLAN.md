# Plan

We will introduce a log-space residual skip in CLG-ODE, align training loss with evaluation, and tighten sampling to short intervals, then run a few low-cost fold0 jobs (user-submitted) to beat the identity baseline within ~6h/fold before scaling up.

## Scope
- In: Residual/identity skip in CLG-ODE, log-domain full MSE term, short-interval sampling emphasis, fast-profile hyperparams, sbatch templates for quick ablations, and documentation/plan updates.
- Out: Data regeneration, directory restructuring, or full 5-fold training until the fast variant beats the baseline.

## Action items
[ ] Inspect `src/engine/clg_trainer.py` and `src/models/clg_ode.py` to identify insertion points for residual skip and full-log MSE loss.
[ ] Implement log-space residual skip: `pred_log = log1p(a0) + s(dt) * tanh(delta_log)` with `s(dt)=dt/(dt+tau)`; add CLI flags in `scripts/train_clg_ode.py`.
[ ] Add a small-weight full-edge log MSE loss to align training with `test_sc_metrics`; add a flag to force `adjacent_pair_prob=1.0`.
[ ] Define a fast profile (target ≤6h/fold): reduce `latent_dim/hidden_dim`, `solver_steps`, `max_epochs`, `patience`, and set `lambda_topo/kl/vel/acc=0`.
[ ] Update `scripts/submit_clg_ode.sh` to accept env overrides for fast-profile flags, and draft 3 sbatch presets for fold0 ablations.
[ ] Run fold0 jobs (user-submitted) and compare `test_sc_metrics.json` vs identity baseline and other baselines; select the best config for later 5-fold.
[ ] Update `docs/workflow.md`, `PROGRESS.md`, and `docs/sessions/` with the baseline-beating strategy and fast-profile settings.

## Open questions
- Default `tau` for the dt-bounded residual scale (e.g., 1.0 year)?
- Preferred range for log-MSE weight (e.g., 0.05 vs 0.1)?
- Should we gate the residual skip by a learnable scalar or keep it fixed to dt-only?
