# Databricks MLOps Workshop — Design Spec

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan
**Branch:** `feat/mlops-workshop`

## 1. Purpose & audience

A ~3-hour, hands-on Databricks workshop teaching the **MLflow model lifecycle**
(tracking → registry → versioning → reproducibility → composition → governance).
The data science is a vehicle, not the goal.

- **Audience:** basic–intermediate Databricks users, low MLOps experience.
- **Teaching device:** one shared synthetic dataset, three deliberately different
  model "shapes", one governed MLflow lifecycle.

### Hard constraint — customer-agnostic
The workshop is published publicly. It must contain **no customer-identifying
details** of any kind: no real customer name, business unit, or proprietary model
name in code, docs, notebooks, comments, or commit messages. A fictional
**"AnyCompany"** is used only where a company name is needed. The domain is
abstracted to a generic **commodity raw-material price-forecasting + monthly
purchase-decision** scenario. Synthetic data only, fixed seed.

## 2. Narrative spine

"One problem, contrasted, converging on a custom model." AnyCompany buys a
commodity raw material monthly and wants to forecast next-month price and decide
how much to purchase.

1. **sklearn** baseline — predict next-month price from ~10 drivers (tabular regression).
2. **PyTorch** deep model — SAME data, SAME target, SAME split, deeper model
   (proves MLflow is flavor-agnostic; serves as the *challenger*).
3. **Pyomo** custom model — consumes the predicted price + monthly economics to make
   the optimal monthly purchase/allocation decision, wrapped as an MLflow custom
   PyFunc (the "weird custom model lives in the same governed lifecycle" payoff).

### Lifecycle stages demonstrated (hands-on)
- **Track & Register** — experiment tracking, params/metrics/artifacts/signatures, UC Model Registry.
- **Evaluate & Validate** — `mlflow.evaluate`, champion–challenger gate with promote-if-wins.
- **Deploy & Serve** — registry-load batch chain (core); Model Serving endpoints (optional lab).

## 3. Deliverables (build order: A, then B)

| # | Deliverable | Location |
|---|---|---|
| A | Databricks `.py` source notebooks | `notebooks/` |
| B | MkDocs workshop website (GH Pages) | `docs/`, `mkdocs.yml`, `.github/workflows/` |

The two are coupled but independently buildable: site lab pages are authored from
this design ("specific placeholder" content), so the site does not block on finished
notebook code. **Build notebooks first**, then author the site documenting the real,
working code.

---

## PART A — Notebooks

### Implementation defaults (with rationale)
- **Notebook format:** Databricks source `.py` (`# Databricks notebook source` +
  `# COMMAND ----------` cells). Git-friendly and importable, unlike `.ipynb`.
- **Shared config:** a `_config.py` notebook `%run` from each lab notebook (widgets
  for `catalog` / `schema`, derived model/experiment names) — Databricks-native, avoids
  per-notebook duplication.
- **Model Registry:** **Unity Catalog**, widget-parameterized
  (`{catalog}.{schema}.<model>`), `mlflow.set_registry_uri("databricks-uc")`. Ships
  publicly with no hardcoded workspace.
- **Pyomo solver:** **HiGHS via `highspy`** (pure-pip, no system binary) through
  Pyomo's `appsi_highs` interface. Reproducible on ML runtime/serverless and
  installable in a serving container.
- **Reproducibility:** fixed random seed; pinned `requirements.txt`; record runtime.
- **Scale:** pandas (toy scale, 96 monthly rows); save dataset as CSV + Delta.

### Notebook layout
```
notebooks/
  _config.py              # widgets (catalog, schema), names, experiment path — %run'd by all
  00_setup.py             # install deps, set registry URI, create UC catalog/schema, set experiment
  01_generate_data.py     # synthetic dataset → CSV + Delta, data dictionary, sanity checks
  02_sklearn_baseline.py  # Model 1: train, log, register v1, set @champion alias
  03_pytorch_challenger.py# Model 2: same data/target/split, MLP, register v2, evaluate vs champion, promote-if-wins
  04_pyomo_pyfunc.py      # Model 3: Pyomo optimization as PyFunc, register purchase_optimizer, load & call
  05_end_to_end.py        # load all from registry by alias, run chain on latest month + UC lineage beat
  99_optional_serving.py  # OPTIONAL: deploy all three to Model Serving endpoints + query
requirements.txt          # pinned: mlflow, scikit-learn, torch, pyomo, highspy, pandas
```

### A.1 — Data (`01_generate_data.py`)
- Monthly time-indexed `month` column (96 months), **time-ordered, never shuffled**
  (forecast framing predicts month t+1 from month t).
- ~10 named drivers with plausible ranges/distributions: `demand_index`,
  `input_cost_index`, `fx_rate`, `inventory_level`, `industrial_output`,
  `energy_cost`, `scrap_supply`, `export_demand`, `seasonality`, `competitor_price`.
- **Causal target:** `price_next_month` = weighted function of drivers + lagged price
  + noise (NOT independent random columns). Tune signal-to-noise so test R² ≈ 0.6–0.85.
- Derived `trend_up` (bool): 1 if `price_next_month > current price`. Optional
  classification variant; models stay regression to keep focus.
- **Decision-economics columns** (predictors ignore; Pyomo needs): `holding_cost`,
  `purchase_cost`, `demand`, `capacity`, `budget` per month.
- Prints: data dictionary (column, type, meaning, range) + sanity checks (head,
  describe, driver↔target correlations, naive "predict last month's price" baseline R²).

### A.2 — sklearn baseline (`02_sklearn_baseline.py`)
- Time-respecting train/test split (cut by time, **no random shuffle**).
- `GradientBoostingRegressor` (or RandomForest) on the ~10 drivers + lagged price.
- Log params, metrics (RMSE, MAE, R²), model, input example + signature, to the
  shared experiment.
- Register to UC as `{catalog}.{schema}.price_forecaster` **v1**; set alias `@champion`.

### A.3 — PyTorch challenger (`03_pytorch_challenger.py`)
- **Same features, target, and split** as A.2 (the parity is the teaching point).
- Small MLP (modest; audience is not DL-focused). Same metrics, same experiment.
- Register as **v2** of `price_forecaster` (same name, different flavor).
- `mlflow.evaluate` on the held-out test set; explicit RMSE comparison vs current
  `@champion`. **Move `@champion` alias to v2 only if it beats the champion.** Note the
  production-grade `validate_evaluation_results` / `MetricThreshold` API as a next step.

### A.4 — Pyomo PyFunc (`04_pyomo_pyfunc.py`)
- **No training.** Wrap a Pyomo optimization as `mlflow.pyfunc.PythonModel`.
- Decision: given predicted next-month price + monthly `holding_cost`,
  `purchase_cost`, `demand`, `capacity`, `budget`, choose optimal monthly purchase /
  allocation quantity to **minimize total cost** s.t. demand met and capacity/budget
  constraints. Solver: HiGHS via `highspy` (`appsi_highs`).
- Log as PyFunc with signature; register as a **separate** model
  `{catalog}.{schema}.purchase_optimizer`. Load from registry and call to prove it
  lives in the same governed lifecycle.

### A.5 — End-to-end (`05_end_to_end.py`)
- Load `price_forecaster@champion` and `purchase_optimizer` from the registry by alias.
- Run the chain on the latest month: `drivers → predicted price → Pyomo decision`.
- **UC governance/lineage beat:** show the lineage (Delta table → experiment →
  registered models) and frame it as "an ungoverned monthly manual run now lives in a
  governed, lineage-tracked lifecycle."

### A.6 — Optional serving (`99_optional_serving.py`)
- Deploy all three to Model Serving endpoints and query them. Clearly marked optional;
  notes the solver-in-container requirement for the Pyomo endpoint (`highspy` is
  pip-installable, which is why it works).

---

## PART B — MkDocs workshop site

Styled like AWS Workshops, Databricks-branded, auto-deployed to GitHub Pages. The site
is the narrative spine; each lab page points at its matching notebook. **Must not touch
`notebooks/` files or the notebooks' `requirements.txt`.**

### B.1 — Project structure
```
mkdocs.yml
requirements-docs.txt          # SEPARATE from notebooks' requirements.txt — do not merge/overwrite
.github/workflows/gh-pages.yml
docs/
  index.md
  stylesheets/custom.css
  assets/                      # logos/images
  setup/prerequisites.md
  lab-1-generate-data/index.md
  lab-2-sklearn-baseline/index.md
  lab-3-pytorch-challenger/index.md
  lab-4-pyomo-optimizer/index.md
  lab-5-end-to-end/index.md
  optional-serving/index.md
specs/                         # internal design docs — EXCLUDED from the built site
```

### B.2 — `requirements-docs.txt`
Pin `mkdocs-material`, `mkdocs-minify-plugin`, and any other plugins used.

### B.3 — `mkdocs.yml` (AWS-Workshops feel)
- Theme: Material for MkDocs.
- Features: `navigation.sections`, `navigation.expand`, `navigation.footer`
  (Next/Previous), `navigation.top`, search.
- **Right-hand TOC:** Material default (do **NOT** use `toc.integrate` — keep the
  separate right-hand TOC, the AWS look).
- Markdown extensions: `pymdownx.highlight`, `pymdownx.superfences` (copy buttons),
  `pymdownx.admonition`, `pymdownx.tabbed` (`alternate_style: true`),
  `pymdownx.details`, `attr_list`.
- `extra_css: [stylesheets/custom.css]`.
- `exclude_docs:` excludes `specs/` (belt-and-suspenders; primary spec location is the
  top-level `specs/` dir, outside `docs/`).
- `nav:` per B.6.

### B.4 — Databricks branding (`docs/stylesheets/custom.css`)
Brand colors: orange/coral `#FF3621`, dark navy/slate `#1B3139`.
- `--md-primary-fg-color: #FF3621` — header bar + primary buttons **only**.
- **Contrast fix (load-bearing):** `--md-accent-fg-color: #1B3139` (or darkened orange
  ~`#C42912`) for links and interactive/hover states. Bright orange must NOT be the
  link/accent color (fails WCAG on white).
- `#1B3139` for header/footer background accents and sidebar active states.
- Code blocks: sleek dark slate derived from `#1B3139`; good contrast on copy buttons.
- Verify: header text legible on orange; body links legible on white in default and
  any slate sections.

### B.5 — Lab pages (1:1 with notebooks)
Each page:
- Opens with a 1–2 sentence "what you'll do / why it matters" tied to the MLflow
  lifecycle story.
- References its matching notebook by name (e.g. "Open `notebooks/02_sklearn_baseline.py`").
- Uses admonitions: `!!! info` (context), `!!! warning` (gotchas), `!!! success`
  (checkpoints).
- Includes ≥1 `pymdownx.tabbed` code block where multiple languages apply (e.g.
  PySpark / SQL).

Per-page content (specific to each notebook's purpose):
- **index.md** — welcome: who AnyCompany is, what attendees build (one dataset → three
  model shapes → one governed MLflow lifecycle), ~3 hours, audience level.
- **setup/prerequisites.md** — required cloud permissions + Unity Catalog access in a
  `!!! warning`; notes `00_setup.py` runs deps + UC catalog/schema + experiment creation.
- **lab-1** — generating the time-indexed synthetic dataset; data dictionary + sanity
  checks (correlations, naive baseline R²).
- **lab-2** — train sklearn baseline, log to MLflow, register v1, set `@champion`.
- **lab-3** — PyTorch MLP on the SAME data/split, evaluate vs champion RMSE, promote
  `@champion` only if it wins — the champion–challenger gate.
- **lab-4** — wrap Pyomo (HiGHS via highspy) as a PyFunc `PythonModel`, register the
  SEPARATE `purchase_optimizer`, load & call.
- **lab-5** — load both registered models by alias, run drivers → predicted price →
  Pyomo decision on the latest month; UC governance/lineage beat with a `!!! success`
  payoff callout.
- **optional-serving** — deploy all three to Model Serving; note the solver-in-container
  requirement for the Pyomo endpoint.

### B.6 — `nav`
```
Home: index.md
Setup:
  - Prerequisites: setup/prerequisites.md
Lab 1 — Generate the dataset: lab-1-generate-data/index.md
Lab 2 — sklearn baseline (champion): lab-2-sklearn-baseline/index.md
Lab 3 — PyTorch challenger: lab-3-pytorch-challenger/index.md
Lab 4 — Pyomo optimizer: lab-4-pyomo-optimizer/index.md
Lab 5 — End-to-end chain: lab-5-end-to-end/index.md
Optional — Model Serving: optional-serving/index.md
```

### B.7 — GitHub Pages deploy (`.github/workflows/gh-pages.yml`)
- Trigger on push to `main`.
- Checkout → setup Python → `pip install -r requirements-docs.txt` →
  `mkdocs gh-deploy --force`.
- Workflow permission `contents: write` so `gh-deploy` can push to `gh-pages`.

## 4. Verification (definition of done)
- **Notebooks:** each runs top-to-bottom on a Databricks ML runtime; data sanity checks
  print; three models register; champion–challenger promotes correctly; end-to-end chain
  produces a purchase decision. (Verified in-workspace; exact verification harness TBD in
  the implementation plan.)
- **Site:** `mkdocs build --strict` passes with zero nav/link/extension warnings;
  contrast verified (header-on-orange, links-on-white); gh-pages workflow valid.

## 5. Out of scope
- Automate & Monitor lifecycle stage (scheduled retraining, inference/lakehouse
  monitoring, CI for the notebooks) — not in this workshop.
- Real customer data or any non-synthetic dataset.
- Production hardening beyond what the teaching narrative requires.
