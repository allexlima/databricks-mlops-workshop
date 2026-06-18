# Databricks MLOps Workshop — Site

This is the **`workshop-site`** branch: the source for the workshop's MkDocs website,
kept deliberately separate from the notebook code (which lives on `main`) so the two
don't get mixed up.

The site is a guided, hands-on walkthrough; each lab page maps 1:1 to a
notebook on `main` and references it by name.

## Build locally

```bash
python3 -m venv .venv-docs
.venv-docs/bin/pip install -r requirements.txt
.venv-docs/bin/mkdocs serve     # live preview at http://127.0.0.1:8000
.venv-docs/bin/mkdocs build --strict   # production build (zero warnings)
```

## Deploy

Pushing to `workshop-site` runs `.github/workflows/gh-pages.yml`, which builds and
publishes to the `gh-pages` branch via `mkdocs gh-deploy`. One-time setup: in the repo
**Settings → Pages**, set the source to the `gh-pages` branch.
