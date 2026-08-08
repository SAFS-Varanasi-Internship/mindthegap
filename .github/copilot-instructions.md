# Copilot Instructions for `mindthegap`

`mindthegap` is a Python package for training and using models that gap-fill
ocean-color data. Reusable code lives in `mindthegap/`, tests in `tests/`, and
pipeline/example notebooks in `book/`.

The codebase is evolving rapidly. **Treat the existing code and tests as the
source of truth for current APIs and architecture.** Inspect relevant
implementations and call sites before making changes rather than relying on
descriptions in this file.

## Development

- Install locally with: `python -m pip install -e .`
- Run the full test suite with: `python -m pytest tests/`
- Prefer running the relevant test file/test while developing, then broader
  tests when appropriate.
- There is no configured linter or formatter; match the surrounding code style.

`environment.yml` is used by the `repo2docker` workflow to build a legacy
JupyterHub image. It is **not** the development environment for the package.
The package currently targets Python >= 3.11.

## Repository conventions

- Public package functions are exposed through `mindthegap/__init__.py`.
  When adding or changing a public API, check whether exports and callers need
  updating.
- Heavy optional dependencies are generally imported lazily so basic package
  functionality does not require the ML/cloud stack. Preserve this unless the
  task requires changing that design.
- Tests should normally be deterministic and network-independent. Use synthetic
  or in-memory data where practical.
- When changing package behavior, inspect and update affected tests and
  notebooks/examples as appropriate.

## Working in this repository

- Inspect existing code, tests, and call sites before deciding how to implement
  a change.
- Prefer changes consistent with the surrounding design rather than introducing
  unnecessary new abstractions.
- Run relevant code/tests to verify changes rather than assuming an API or
  behavior works.
- If the requested change conflicts with an existing design assumption, favor
  the requested change and update affected code consistently rather than
  preserving stale behavior solely because it exists today.