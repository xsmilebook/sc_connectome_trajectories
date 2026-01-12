# Plan

We will migrate the Git repository boundary from `src/` to the project root so the repository matches `ARCHITECTURE.md`, while keeping runtime artifacts untracked via `.gitignore`.

## Scope
- In: Make the project root the Git repo; keep the stable layout in `ARCHITECTURE.md`; ensure `.gitignore` prevents committing runtime artifacts.
- Out: Refactor scientific code/pipelines beyond what is needed for the repo migration; change dataset semantics; modify runtime artifacts under `data/` or `outputs/`.

## Action items
[x] Tag the pre-migration `src/` repo and create a full backup bundle (`src-pre-root-migration-20260112.bundle`).
[x] Initialize a new Git repo at the project root and commit existing root documentation (excluding runtime artifacts).
[x] Convert `src/` into a normal directory by removing `src/.git`, then commit the current `src/` snapshot into the root repo.
[x] Set the root repo default branch to `main` and restore `origin` remote.
[x] Preserve full `src/` history inside the root repo (imported via `git-filter-repo` rewrite + merge).
[ ] Run validation: `python -m pytest` (if applicable) and a smoke import/execution of key entrypoints from the new root.
[x] Record the migration in `PROGRESS.md` and `docs/sessions/` and document rollback using the bundle.

## Open questions
- None (history is preserved in the bundle; no additional ignore rules requested).
