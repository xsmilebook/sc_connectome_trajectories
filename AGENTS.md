# AGENTS.md

Operational rules for AI-assisted work in `sc_connectome_trajectories`.
This repository treats `ARCHITECTURE.md` as the source of truth for the stable layout. Do not propose structural changes unless explicitly requested.

## Base workflow

1) Read `ARCHITECTURE.md` before planning or editing.
2) Execute changes only according to `PLAN.md` or explicit user instructions.
3) If a change affects usage or structure, update `README.md` and/or `docs/workflow.md`.
4) Update `PROGRESS.md` after completing a change set.
5) Log each session in `docs/sessions/` (date-stamped file).
6) If you modify code or documentation, commit the changes with git (write your own message).
7) If the `create-plan` skill is used, write the plan to the root `PLAN.md`.

## Scope and constraints

- Default scope is documentation-only unless the user requests engineering changes.
- Do not modify runtime artifacts under `data/` or `outputs/`.
- Do not change directory names or structure unless explicitly requested.
- Keep diffs minimal; avoid refactors not tied to the request.

## Engineering conventions (when requested)

- Use `python -m scripts.<entry>` for execution.
- Avoid ad-hoc `sys.path` hacks; if unavoidable, confine them to entry points and document why.
- All filesystem paths must come from `configs/`.

## Documentation conventions

- Use precise, scientific language.
- Write all `docs/` content in Chinese; keep filenames in English.
- Root-level Markdown files must be English only.
- Keep repository root folder names in English.
- sc_connectome_trajectories-specific notes live in `docs/workflow.md` and `configs/paths.yaml` (dataset section).
