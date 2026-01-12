# Plan

We will migrate the Git repository boundary from `src/` to the project root so the repository matches `ARCHITECTURE.md`, while keeping runtime artifacts untracked via `.gitignore`.

## Scope
- In: Make the project root the Git repo; optionally preserve `src/` history; keep the stable layout in `ARCHITECTURE.md`; ensure `.gitignore` prevents committing runtime artifacts.
- Out: Refactor scientific code/pipelines beyond what is needed for the repo migration; change dataset semantics; modify runtime artifacts under `data/` or `outputs/`.

## Action items
[ ] Confirm whether `src/` Git history must be preserved (full history vs squashed import vs fresh start).
[ ] Create a safety backup of the current `src` repo (`git -C src bundle create ..\\src.git.bundle --all`).
[ ] Initialize a new Git repo at the project root and commit existing root documentation/config scaffolding (excluding ignored runtime artifacts).
[ ] Import the existing `src` repo into the root repo under the `src/` prefix (prefer `git subtree add --prefix=src <path-to-src> main`; alternative: `git filter-repo --to-subdirectory-filter src`).
[ ] Reconnect remotes/branches/tags and verify `git log`, `git status`, and that `.gitignore` keeps `data/`, `outputs/`, `notebooks/`, `models/`, `tmp/`, and logs untracked.
[ ] Remove the nested repo marker (`src/.git`) and ensure `src/` is a normal directory within the root repo.
[ ] Align the top-level directory set with `ARCHITECTURE.md` (create missing folders only when needed; keep runtime folders like `outputs/` ignored).
[ ] Update `README.md` and `docs/workflow.md` to reflect the new repo root, standard entrypoints (`python -m scripts.<entry>`), and any path/config expectations.
[ ] Run validation: `python -m pytest` (if applicable) and a smoke import/execution of key entrypoints from the new root.
[ ] Add rollback instructions (restore from `src.git.bundle`) and record the migration in `PROGRESS.md` and `docs/sessions/`.

## Open questions
- Should we preserve `src/` commit history exactly, or is a squashed import acceptable?
- Should the existing root `README.md` replace the historical `src/README.md`, or should we merge them?
- Are there additional ignore rules you want beyond the current `.gitignore` (e.g., `.venv/`, `*.pt`, `*.ckpt`)?
