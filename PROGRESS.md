# PROGRESS.md


## Current focus
- 2026-01-12: Make the project root the Git repository root (migrate from `src/`).

## Completed
- 2026-01-12: Drafted the repo-structure refactor plan in `PLAN.md`.
- 2026-01-12: Tagged and backed up the original `src/` repo as `src-pre-root-migration-20260112.bundle`.
- 2026-01-12: Initialized a new root Git repo and committed root docs + current `src/` snapshot.
- 2026-01-12: Removed nested `src/.git` so `src/` is now a normal directory in the root repo.
- 2026-01-12: Reorganized runtime data into the single-dataset layout (`data/interim/`, `data/processed/`) and moved figures/results under `outputs/`.
- 2026-01-12: Imported the full legacy `src` commit history into the root repo (rewritten under the `src/` prefix).
- 2026-01-12: Added root-level `scripts/` entrypoints and made `src/` importable as a package.

## In progress

## Next up
- Push the new root repo to `origin` (may require forced push depending on the remote state).
- Decide whether to import full historical commits from the old `src/` repo into the root repo (optional; bundle available).

## Issues and solutions

| Issue | Solution | Date |
|-------|----------|------|

## Notes
