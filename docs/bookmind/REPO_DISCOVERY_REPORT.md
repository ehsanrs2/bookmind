# Repo Discovery Report (Bookmind)

Generated: 2026-01-26

## 1) High-level structure

Top-level directories:
- `backend/` — Python backend package and related assets.
- `src/` — SvelteKit frontend source (Svelte + TypeScript).
- `static/` — Static frontend assets, pyodide bundle, themes, manifest.
- `docs/` — Markdown documentation.
- `scripts/` — Project scripts (e.g., build helpers).
- `cypress/` — Cypress E2E tests and fixtures.
- `test/` — Test fixtures (non-code) used by backend tests.
- `.github/` — Issue templates only (no workflows).

Backend location/language/framework:
- `backend/open_webui/` is the primary Python package; dependencies indicate FastAPI + Uvicorn.

Frontend location/language/framework:
- `src/` contains SvelteKit app (Svelte 5 + Vite, TypeScript).

## 2) Build & tooling

Package managers / dependency management:
- Node: `package.json` + `package-lock.json` (npm).
- Python: `pyproject.toml` (Hatch build), plus `uv.lock`; also `backend/requirements.txt` and `backend/requirements-min.txt` are present.
- Docker: `Dockerfile` and multiple `docker-compose*.yaml` files.

Lint / format tooling:
- ESLint: `.eslintrc.cjs` (Svelte + TypeScript + Cypress). `npm run lint:frontend`.
- Prettier: `.prettierrc` + `.prettierignore`. `npm run format`.
- Pylint: `npm run lint:backend` uses `pylint backend/` (no project-level config file found).
- Black: `npm run format:backend` runs `black . --exclude ".venv/|/venv/"` (no `tool.black` config in `pyproject.toml`).
- Codespell config exists in `pyproject.toml`.
- Tailwind/PostCSS config: `tailwind.config.js`, `postcss.config.js`.

Testing:
- Python: `pytest` (tests under `backend/open_webui/test/`).
- Frontend unit: `vitest` (`npm run test:frontend`).
- E2E: `cypress/` with `cypress.config.ts` (run via `cypress open`).
- Playwright is referenced in optional Python deps and docker-compose, but no local config file found.

## 3) Docs conventions

Docs locations:
- `README.md` at repo root.
- `docs/` directory contains Markdown (`README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `apache.md`).

Conventions:
- No explicit docs system (no MkDocs/Docusaurus configs found).
- Docs appear to be plain Markdown files in `docs/`.

## 4) Python environment specifics

Dependencies:
- Primary: `pyproject.toml` (Hatch build, dependency list in `[project]`).
- Also present: `backend/requirements.txt` and `backend/requirements-min.txt`.
- Lockfile: `uv.lock` (suggests uv/rye usage).

Module layout:
- Python package root: `backend/open_webui/`.
- Common utility modules live in `backend/open_webui/utils/`.
- Tests live in `backend/open_webui/test/` (e.g., `util/`, `apps/`).

Tests (how to run):
- Likely `pytest backend/open_webui/test` (no Makefile/test script defined).

Safe place for small internal utilities:
- `backend/open_webui/utils/` is the established location for reusable helpers.

## 5) Existing schema/spec patterns

- No static JSON Schema or OpenAPI spec files found in the repo (no `*.schema.json`, `openapi*.yml`, etc.).
- OpenAPI appears to be generated dynamically by FastAPI; related tooling exists in `backend/open_webui/utils/tools.py` and frontend helper code in `src/lib/utils/index.ts`.
- Swagger UI assets are present in `backend/open_webui/static/swagger-ui/`.

## 6) Suggested placement for Bookmind additions

A) Scope schema + chunk metadata schema (JSON Schema)
- Option 1 (backend-first): `backend/open_webui/schemas/bookmind/` (new dir).
  - Pros: close to backend code; easier to load at runtime for validation.
  - Cons: no existing `schemas/` convention.
- Option 2 (docs-first): `docs/bookmind/schemas/`.
  - Pros: aligns with documentation use; avoids backend coupling.
  - Cons: less convenient for runtime validation.

B) Example scope JSONs
- Option 1: `docs/bookmind/examples/`.
  - Pros: co-located with spec docs.
  - Cons: examples not directly used in tests.
- Option 2: `test/test_files/bookmind/`.
  - Pros: aligns with existing test fixtures.
  - Cons: less discoverable for documentation readers.

C) Small Python validator module + tests
- Option 1: `backend/open_webui/utils/bookmind_validator.py` with tests in `backend/open_webui/test/util/`.
  - Pros: consistent with existing `utils/` layout; tests already grouped by util.
  - Cons: `utils/` is broad; no validators subpackage.
- Option 2: `backend/open_webui/internal/bookmind/validator.py` with tests in `backend/open_webui/test/`.
  - Pros: keeps Bookmind-specific code isolated.
  - Cons: `internal/` is used for other infra (migrations/db) and may be less appropriate for reusable helpers.

D) Documentation for Bookmind specs
- Option 1: `docs/bookmind/` (proposed home), with an index file such as `docs/bookmind/README.md`.
  - Pros: aligns with current docs pattern; easy to browse.
- Option 2: `docs/` root with Bookmind-prefixed filenames (e.g., `docs/bookmind-schemas.md`).
  - Pros: avoids adding a subfolder.
  - Cons: loses grouping and organization.

## 7) GitHub workflows status

- `.github/workflows/` is absent; only `.github/ISSUE_TEMPLATE/` exists. This means there is currently no automated CI on push/PR.

Minimal future CI plan (text-only suggestion):
- Add a single workflow that runs on `push` and `pull_request` with:
  - Node job: `npm ci`, `npm run lint:frontend`, `npm run test:frontend`.
  - Python job: install deps (pyproject or requirements), `pytest backend/open_webui/test`.
  - Optional: `black --check .` and `pylint backend/`.

