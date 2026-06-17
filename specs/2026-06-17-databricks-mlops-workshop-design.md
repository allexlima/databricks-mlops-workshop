# Databricks MLOps Workshop — Design Spec

**Date:** 2026-06-17
**Status:** Approved design, pending implementation plan
**Branch:** `feat/mlops-workshop`
**Rev:** 2 — single canonical predictor (sklearn) + Pyomo headline custom path;
PyTorch demoted to optional; champion–challenger replaced by a deterministic
single-model validation gate.

## 1. Purpose & audience

A ~3-hour, hands-on Databricks workshop teaching the **MLflow model lifecycle**
(tracking → registry → versioning → reproducibility → composition → governance).
The data science is a vehicle, not the goal.

- **Audience:** basic–intermediate Databricks users, low MLOps experience.
- **Teaching device:** one shared synthetic dataset, two model shapes on one governed
  MLflow lifecycle — a **standard ML predictor (sklearn)** and an **unusual custom OR
  model (Pyomo)** — with a third framework (PyTorch) shown as an optional "same
  lifecycle, different flavor" interlude.
- **Teaching claim:** *MLflow governs whatever you've got — including an unusual
  operations-research model* — not merely "MLflow supports many frameworks." This
  mirrors the real customer situation (a custom OR model run manually each month).

### Hard constraint — customer-agnostic
The workshop is published publicly. It must contain **no customer-identifying
details** of any kind: no real customer name, business unit, or proprietary model
name in code, docs, notebooks, comments, or commit messages. A fictional
**"AnyCompany"** is used only where a company name is needed. The domain is
abstracted to a generic **commodity raw-material price-forecasting + monthly
purchase-decision** scenario. Synthetic data only, fixed seed.

## 2. Narrative spine — two acts

"One problem, two paths, one governed lifecycle." AnyCompany buys a commodity raw
material monthly and wants to forecast next-month price and decide how much to purchase.

- **Act 1 — Standard ML path (sklearn).** Predict next-month price from ~10 drivers
  (tabular regression). Train → track → evaluate against a fixed quality bar → register
  and promote to `@champion` only if it passes. The everyday MLOps loop.
- **Act 2 — Custom path (Pyomo), the headline.** A Pyomo optimization that consumes the
  predicted price + monthly economics to make the optimal monthly purchase/allocation
  decision, wrapped as an MLflow custom PyFunc. The "your weird custom OR model lives in
  the exact same governed lifecycle" payoff.
- **Optional interlude — PyTorch.** The same data/target/split through a small MLP,
  framed "same lifecycle, different flavor." It is **not** a challenger and **not** a
  dependency of anything; it reuses the same validation gate, teaching the concept a
  second time for free.

### Lifecycle stages demonstrated (hands-on)
- **Track & Register** — experiment tracking, params/metrics/artifacts/signatures, UC Model Registry.
- **Evaluate & Validate** — `mlflow.evaluate` + a **deterministic single-model
  validation gate**: a model is registered and promoted to `@champion` only if it clears
  a fixed threshold (R² ≥ 0.6). No second model required; "models don't ship unless they
  pass," fully reproducible.
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
  for `catalog` / `schema`, derived model/experiment names, **the random seed as one
  named constant**, and the R² threshold) — Databricks-native, avoids duplication and
  keeps reproducibility knobs in one place.
- **Model Registry:** **Unity Catalog**, widget-parameterized
  (`{catalog}.{schema}.<model>`), `mlflow.set_registry_uri("databricks-uc")`. Ships
  publicly with no hardcoded workspace.
- **Pyomo solver:** **HiGHS via `highspy`** (pure-pip, no system binary) through
  Pyomo's `appsi_highs` interface. Reproducible on ML runtime/serverless and
  installable in a serving container.
- **Reproducibility:** single named seed in `_config.py`; pinned `requirements.txt`;
  record runtime.
- **Scale:** pandas (toy scale, 96 monthly rows); save dataset as CSV + Delta.
- **Alias-first, version-agnostic:** all cross-notebook references use the
  `@champion` alias, **never literal version integers** — re-runs against the same UC
  registry bump version numbers, so any "v2" assumption would break.

### Notebook layout
```
notebooks/
  _config.py              # widgets (catalog, schema), names, experiment path, SEED, R2_THRESHOLD — %run'd by all
  00_setup.py             # install deps, set registry URI, create UC catalog/schema, set experiment
  01_generate_data.py     # synthetic dataset → CSV + Delta, data dictionary, sanity checks, R² band assertion
  02_sklearn_baseline.py  # Standard path: train, evaluate, validation gate, register + @champion if it passes
  04_pyomo_pyfunc.py      # Headline custom path: Pyomo optimization as PyFunc, register purchase_optimizer, load & call
  05_end_to_end.py        # load price_forecaster@champion + purchase_optimizer, run chain on latest month + lineage beat
  97_optional_pytorch.py  # OPTIONAL: same data/split MLP, reuses the same validation gate; NOT a dependency of 05
  99_optional_serving.py  # OPTIONAL: deploy models to Model Serving endpoints + query
requirements.txt          # pinned: mlflow, scikit-learn, torch, pyomo, highspy, pandas
```
(The `03_` slot is intentionally left empty: PyTorch moved out of the main sequence to
`97_`. The mandatory path is 01 → 02 → 04 → 05.)

### A.1 — Data (`01_generate_data.py`)
- Monthly time-indexed `month` column (96 months), **time-ordered, never shuffled**
  (forecast framing predicts month t+1 from month t).
- ~10 named drivers with plausible ranges/distributions: `demand_index`,
  `input_cost_index`, `fx_rate`, `inventory_level`, `industrial_output`,
  `energy_cost`, `scrap_supply`, `export_demand`, `seasonality`, `competitor_price`.
- **Causal target:** `price_next_month` = weighted function of drivers + lagged price
  + noise (NOT independent random columns). Tune signal-to-noise so test R² ≈ 0.6–0.85.
- **R² band assertion:** the generator fits the naive/quick reference fit and
  **asserts the achievable R² lands in [0.6, 0.85], failing loudly otherwise** — the
  demo is dead if the signal is too weak (unbeatable) or too strong (trivial).
- Derived `trend_up` (bool): 1 if `price_next_month > current price`. **Illustration
  only** — kept as a column, never modeled (regression-only workshop, no classification fork).
- **Decision-economics columns** (predictors ignore; Pyomo needs): `holding_cost`,
  `purchase_cost`, `demand`, `capacity`, `budget` per month.
- Prints: data dictionary (column, type, meaning, range) + sanity checks (head,
  describe, driver↔target correlations, naive "predict last month's price" baseline R²).

### A.2 — sklearn standard path (`02_sklearn_baseline.py`)
- Time-respecting train/test split (cut by time, **no random shuffle**).
- `GradientBoostingRegressor` (or RandomForest) on the ~10 drivers + lagged price.
- Log params, metrics (RMSE, MAE, R²), model, input example + signature, to the
  shared experiment.
- **Validation gate:** run `mlflow.evaluate` on the held-out test set; **register to UC
  as `{catalog}.{schema}.price_forecaster` and set alias `@champion` only if test
  R² ≥ `R2_THRESHOLD` (0.6).** If it fails, do not promote and surface why. (Note the
  production-grade `validate_evaluation_results` / `MetricThreshold` API as a next step.)

### A.3 — Pyomo PyFunc (`04_pyomo_pyfunc.py`) — headline
- **No training.** Wrap a Pyomo optimization as `mlflow.pyfunc.PythonModel`.
- Decision: given predicted next-month price + monthly `holding_cost`,
  `purchase_cost`, `demand`, `capacity`, `budget`, choose optimal monthly purchase /
  allocation quantity to **minimize total cost** s.t. demand met and capacity/budget
  constraints. Solver: HiGHS via `highspy` (`appsi_highs`).
- Log as PyFunc with signature; register as a **separate** model
  `{catalog}.{schema}.purchase_optimizer`. Load from registry and call to prove it
  lives in the same governed lifecycle.

### A.4 — End-to-end (`05_end_to_end.py`)
- Load `price_forecaster@champion` and `purchase_optimizer` from the registry **by
  alias** (never by version integer).
- Run the chain on the latest month: `drivers → predicted price → Pyomo decision`.
- **Depends only on the sklearn lab** for `@champion`. Skipping the optional PyTorch lab
  cannot break the finale.
- **UC governance/lineage beat:** show the lineage (Delta table → experiment →
  registered models) and frame it as "an ungoverned monthly manual run now lives in a
  governed, lineage-tracked lifecycle."

### A.5 — Optional PyTorch (`97_optional_pytorch.py`)
- **Same features, target, and split** as A.2. Small MLP (modest; audience is not
  DL-focused). Same metrics, same experiment.
- Reuses the **same validation gate** (R² ≥ threshold → register a new version of
  `price_forecaster`). Framed "same lifecycle, different flavor," **not** a challenger.
- **Not a dependency** of any other notebook. The end-to-end finale ignores it.

### A.6 — Optional serving (`99_optional_serving.py`)
- Deploy models to Model Serving endpoints and query them. Clearly marked optional;
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
  lab-3-pyomo-optimizer/index.md
  lab-4-end-to-end/index.md
  optional/pytorch/index.md
  optional/serving/index.md
specs/                         # internal design docs — OUTSIDE docs/, excluded from the built site
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

### B.5 — Lab pages (mapped to notebooks)
Each page:
- Opens with a 1–2 sentence "what you'll do / why it matters" tied to the MLflow
  lifecycle story.
- References its matching notebook by name (e.g. "Open `notebooks/02_sklearn_baseline.py`").
- Uses admonitions: `!!! info` (context), `!!! warning` (gotchas), `!!! success`
  (checkpoints).
- Includes ≥1 `pymdownx.tabbed` code block where multiple languages apply (e.g.
  PySpark / SQL).

Per-page content (specific to each notebook's purpose):
- **index.md** — welcome: who AnyCompany is, what attendees build (one dataset → a
  standard ML predictor + an unusual custom OR model → one governed MLflow lifecycle),
  ~3 hours, audience level; states the teaching claim ("MLflow governs whatever you've got").
- **setup/prerequisites.md** — required cloud permissions + Unity Catalog access in a
  `!!! warning`; notes `00_setup.py` runs deps + UC catalog/schema + experiment creation.
- **lab-1** (→ `01_generate_data.py`) — generating the time-indexed synthetic dataset;
  data dictionary + sanity checks (correlations, naive baseline R²); the R² band assertion.
- **lab-2** (→ `02_sklearn_baseline.py`) — standard ML path: train sklearn, log to
  MLflow, `mlflow.evaluate`, **validation gate** (register + `@champion` only if
  R² ≥ 0.6).
- **lab-3** (→ `04_pyomo_pyfunc.py`) — **centerpiece**: wrap Pyomo (HiGHS via highspy)
  as a PyFunc `PythonModel`, register the SEPARATE `purchase_optimizer`, load & call.
- **lab-4** (→ `05_end_to_end.py`) — load `price_forecaster@champion` + `purchase_optimizer`
  by alias, run drivers → predicted price → Pyomo decision on the latest month; UC
  governance/lineage beat with a `!!! success` payoff callout.
- **optional/pytorch** (→ `97_optional_pytorch.py`) — same data/split through an MLP,
  "same lifecycle, different flavor," reuses the validation gate; explicitly optional and
  not required for the finale.
- **optional/serving** (→ `99_optional_serving.py`) — deploy to Model Serving; note the
  solver-in-container requirement for the Pyomo endpoint.

### B.6 — `nav`
```
Home: index.md
Setup:
  - Prerequisites: setup/prerequisites.md
Lab 1 — Generate the dataset: lab-1-generate-data/index.md
Lab 2 — sklearn standard path: lab-2-sklearn-baseline/index.md
Lab 3 — Pyomo optimizer (headline): lab-3-pyomo-optimizer/index.md
Lab 4 — End-to-end chain: lab-4-end-to-end/index.md
Optional:
  - PyTorch — same lifecycle, different flavor: optional/pytorch/index.md
  - Model Serving: optional/serving/index.md
```

### B.7 — GitHub Pages deploy (`.github/workflows/gh-pages.yml`)
- Trigger on push to `main`.
- Checkout → setup Python → `pip install -r requirements-docs.txt` →
  `mkdocs gh-deploy --force`.
- Workflow permission `contents: write` so `gh-deploy` can push to `gh-pages`.

## 4. Verification (definition of done)

### Notebooks — minimum bar (workspace-agnostic; run manually)
A `%run`-able verification cell/notebook executes the mandatory path **01 → 02 → 04 →
05** and asserts four checkpoints:
1. **R² in band** — generated-data achievable R² ∈ [0.6, 0.85].
2. **Model registered** — `price_forecaster` exists in the UC registry.
3. **`@champion` resolves** — the `@champion` alias points to a registered version
   (assert on the **alias**, never on a literal version integer — re-runs bump versions).
4. **Chain returns a decision** — the end-to-end chain produces a numeric purchase
   decision for the latest month.

Does not assume a specific workspace; the author runs the notebooks manually against
their own UC catalog/schema via the widgets.

### Site
- `mkdocs build --strict` passes with zero nav/link/extension warnings.
- Contrast verified (header-on-orange, links-on-white in default and slate sections).
- gh-pages workflow valid.

## 5. Out of scope
- Automate & Monitor lifecycle stage (scheduled retraining, inference/lakehouse
  monitoring, CI for the notebooks) — not in this workshop.
- Champion–challenger / multi-model promotion (replaced by the deterministic
  single-model gate).
- A classification fork on `trend_up` (regression-only).
- Real customer data or any non-synthetic dataset.
- Production hardening beyond what the teaching narrative requires.
