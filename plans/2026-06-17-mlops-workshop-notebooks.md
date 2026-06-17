# MLOps Workshop — Notebooks Implementation Plan

> **Execution:** Inline, notebook-by-notebook, with a checkpoint after each notebook so the author can run it in-workspace before continuing. No test suite — verification is **inline asserts inside the notebooks** plus `06_verify` (the four-checkpoint run-and-see-green harness). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Databricks `.py` source notebooks for an MLOps workshop where one shared synthetic dataset flows through a standard ML predictor (sklearn) and an unusual custom OR model (Pyomo), all governed by one MLflow lifecycle on Unity Catalog.

**Architecture:** Pure, reusable logic (synthetic data generation, the Pyomo purchase optimizer, the PyFunc wrapper) lives in a small importable module `notebooks/workshop_lib.py` so notebooks **import** it rather than duplicating logic. The notebooks are thin orchestration layers that add the MLflow/Unity-Catalog/registry/serving glue. Correctness is enforced by **inline asserts that fail loudly when a cell is run** (R²-in-band in `01`, optimizer feasibility in `04`) and by `06_verify`. `_config.py` centralizes widgets and the reproducibility constants.

**Tech Stack:** Python, pandas, NumPy, scikit-learn, Pyomo + HiGHS (`highspy`, `appsi_highs`), PyTorch (optional lab, CPU-only), MLflow **3.x** (UC registry), Databricks notebooks (`.py` source format).

## Global Constraints

- **Customer-agnostic:** no real customer/business-unit/model names anywhere (code, docs, comments, commits). Use fictional **"AnyCompany"** only if a name is needed. Synthetic data only.
- **No test suite:** do NOT create `tests/`, `pytest.ini`, or a test runner. Verification is inline asserts in the notebooks + `06_verify`.
- **Notebook format:** Databricks source `.py` — first line `# Databricks notebook source`, cells separated by `# COMMAND ----------`, markdown via `# MAGIC %md`.
- **MLflow 3.x:** `log_model(name=..., python_model=...)` — use `name=`, **not** the deprecated `artifact_path=`. `PythonModel.predict` signature is `predict(self, context, model_input, params=None)`.
- **Registry:** Unity Catalog (`mlflow.set_registry_uri("databricks-uc")`), names `{catalog}.{schema}.<model>`, widget-parameterized — no hardcoded workspace.
- **Alias-first:** all model loads use the `@champion` alias — **never** literal version integers, **never** `/latest`.
- **Solver:** HiGHS via `highspy` through Pyomo `appsi_highs` (pure pip, no system binary).
- **Reproducibility:** single named `SEED`; achievable test R² must land in **[0.6, 0.85]**; promotion gate threshold `R2_THRESHOLD = 0.6`.
- **Regression-only:** `trend_up` is a derived illustration column, never modeled.
- **Mandatory notebook path:** `01 → 02 → 04 → 05`. PyTorch (`97_`) and serving (`99_`) are optional and must not be dependencies of the finale.
- **Do not** create or modify the MkDocs site files (`docs/`, `mkdocs.yml`, `.github/`) — separate plan.

## Already done (commits on `feat/mlops-workshop`)
- `requirements.txt` with MLflow-3 / CPU-only-torch pins; `notebooks/.gitkeep`. (`ae70ce8`, `ed0f7e6`)

---

### Task A: `workshop_lib.py` — pure reusable logic (no tests)

**Files:**
- Create: `notebooks/workshop_lib.py`

**Produces (imported by the notebooks):**
- Constants `SEED: int`, `R2_THRESHOLD = 0.6`, `R2_BAND = (0.6, 0.85)`, `DRIVERS: list[str]` (10 names), `ECON_COLS = ["holding_cost","purchase_cost","demand","capacity","budget"]`.
- `generate_dataset(n_months=96, seed=SEED) -> pd.DataFrame` — time-ordered monthly frame: `month` (0..n-1, monotonic), 10 `DRIVERS`, `price`, `price_next_month`, `trend_up` (bool), 5 `ECON_COLS`.
- `quick_fit_r2(df) -> float` — time-ordered 80/20 split, small GBR on `DRIVERS+["price"]`, returns test R².
- `solve_purchase(predicted_price, holding_cost, purchase_cost, demand, capacity, budget) -> dict` → `{"purchase_qty","total_cost","status"}` (`"optimal"`/`"infeasible"`).
- `PurchaseOptimizerModel(mlflow.pyfunc.PythonModel)` — `predict(self, context, model_input, params=None)` reads `["predicted_price",*ECON_COLS]` row-wise, returns frame `["purchase_qty","total_cost","status"]`.

- [ ] **Step 1: Write `notebooks/workshop_lib.py`**

```python
"""Reusable, framework-agnostic logic for the MLOps workshop.

Imported (never copy-pasted) by the Databricks notebooks so they stay thin and
DRY. Pure pandas/NumPy/Pyomo, plus mlflow only for the PyFunc base class — no
Databricks or dbutils dependency.

Customer-agnostic: contains no real customer, business-unit, or proprietary model
names. Generic synthetic commodity domain only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow.pyfunc

# --- Reproducibility & gate constants (single source of truth) ---------------
SEED: int = 42
R2_THRESHOLD: float = 0.6                       # promote to @champion only if test R2 >= this
R2_BAND: tuple[float, float] = (0.6, 0.85)      # generator must keep achievable R2 here

DRIVERS: list[str] = [
    "demand_index", "input_cost_index", "fx_rate", "inventory_level",
    "industrial_output", "energy_cost", "scrap_supply", "export_demand",
    "seasonality", "competitor_price",
]
ECON_COLS: list[str] = ["holding_cost", "purchase_cost", "demand", "capacity", "budget"]


def generate_dataset(n_months: int = 96, seed: int = SEED) -> pd.DataFrame:
    """Time-ordered monthly synthetic commodity dataset with a causal target.

    ``price_next_month`` is a weighted function of the drivers + lagged price +
    noise, so an ML model beats a naive baseline but not perfectly (R2 in R2_BAND).
    Rows are time-ordered and never shuffled.
    """
    rng = np.random.default_rng(seed)
    n = n_months
    drivers = {
        "demand_index":      100 + np.cumsum(rng.normal(0, 2, n)),
        "input_cost_index":  100 + np.cumsum(rng.normal(0, 1.5, n)),
        "fx_rate":           1.0 + np.cumsum(rng.normal(0, 0.01, n)),
        "inventory_level":   rng.uniform(50, 150, n),
        "industrial_output": 100 + np.cumsum(rng.normal(0, 1.8, n)),
        "energy_cost":       60 + np.cumsum(rng.normal(0, 1.2, n)),
        "scrap_supply":      rng.uniform(80, 120, n),
        "export_demand":     rng.uniform(40, 90, n),
        "seasonality":       10 * np.sin(2 * np.pi * np.arange(n) / 12),
        "competitor_price":  200 + np.cumsum(rng.normal(0, 2.5, n)),
    }
    df = pd.DataFrame(drivers)
    df.insert(0, "month", np.arange(n))

    z = (df[DRIVERS] - df[DRIVERS].mean()) / df[DRIVERS].std(ddof=0)
    weights = np.array([8, 6, 5, -3, 4, 5, -4, 3, 2, 7], dtype=float)
    base = 200.0 + z.to_numpy() @ weights
    df["price"] = base + rng.normal(0, 6, n)

    signal = 0.6 * df["price"].to_numpy() + 0.4 * base
    noise = rng.normal(0, 14, n)                 # tuned so quick_fit_r2 lands in band
    price_next = signal + noise
    price_next[:-1] = price_next[1:]             # row t predicts t+1
    df["price_next_month"] = price_next
    df["trend_up"] = df["price_next_month"] > df["price"]

    df["holding_cost"]  = rng.uniform(0.5, 2.0, n)
    df["purchase_cost"] = df["price"] * rng.uniform(0.9, 1.1, n)
    df["demand"]        = rng.uniform(80, 140, n)
    df["capacity"]      = rng.uniform(150, 220, n)
    df["budget"]        = rng.uniform(20000, 40000, n)
    return df


def quick_fit_r2(df: pd.DataFrame) -> float:
    """Time-ordered 80/20 split, small GBR, return test R2 (signal sanity check)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score

    feats = DRIVERS + ["price"]
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]
    model = GradientBoostingRegressor(random_state=SEED)
    model.fit(train[feats], train["price_next_month"])
    return float(r2_score(test["price_next_month"], model.predict(test[feats])))


def solve_purchase(predicted_price: float, holding_cost: float, purchase_cost: float,
                   demand: float, capacity: float, budget: float) -> dict:
    """Single-period purchase decision: minimize purchase + holding cost s.t. meeting
    demand within capacity and budget. Solved with HiGHS via Pyomo appsi."""
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers.highs import Highs

    m = pyo.ConcreteModel()
    m.q = pyo.Var(domain=pyo.NonNegativeReals)
    m.leftover = pyo.Var(domain=pyo.NonNegativeReals)
    m.meet_demand = pyo.Constraint(expr=m.q >= demand)
    m.capacity = pyo.Constraint(expr=m.q <= capacity)
    m.budget = pyo.Constraint(expr=purchase_cost * m.q <= budget)
    m.leftover_def = pyo.Constraint(expr=m.leftover >= m.q - demand)
    m.obj = pyo.Objective(expr=purchase_cost * m.q + holding_cost * m.leftover,
                          sense=pyo.minimize)

    result = Highs().solve(m)
    status = str(result.termination_condition).lower()
    if "optimal" not in status:
        return {"purchase_qty": float("nan"), "total_cost": float("nan"), "status": "infeasible"}
    return {"purchase_qty": float(pyo.value(m.q)),
            "total_cost": float(pyo.value(m.obj)), "status": "optimal"}


class PurchaseOptimizerModel(mlflow.pyfunc.PythonModel):
    """Wraps the Pyomo optimizer as an MLflow PyFunc so a non-trainable OR model
    lives in the same registry/lifecycle as the ML predictor."""

    def predict(self, context, model_input, params=None):
        rows = [
            solve_purchase(
                predicted_price=r["predicted_price"], holding_cost=r["holding_cost"],
                purchase_cost=r["purchase_cost"], demand=r["demand"],
                capacity=r["capacity"], budget=r["budget"],
            )
            for _, r in model_input.iterrows()
        ]
        return pd.DataFrame(rows, columns=["purchase_qty", "total_cost", "status"])
```

- [ ] **Step 2: Dev-time sanity check (NOT committed — no test file)**

Run a throwaway snippet against the local `.venv` to confirm the logic before handoff. If R² is outside the band, adjust the `noise` std and re-run until it lands in [0.6, 0.85].
```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'notebooks'); import workshop_lib as wl; \
df=wl.generate_dataset(); r=wl.quick_fit_r2(df); print('R2', round(r,3), 'in band', wl.R2_BAND[0]<=r<=wl.R2_BAND[1]); \
print(wl.solve_purchase(100,5.0,10,100,200,5000)); print(wl.solve_purchase(100,0.1,10,100,200,500))"
```
Expected: `R2 0.6–0.85 in band True`; first solve `status=optimal` with `purchase_qty≈100`; second solve `status=infeasible` (budget 500 < 100×10).

- [ ] **Step 3: Commit**
```bash
git add notebooks/workshop_lib.py
git commit -m "Add workshop_lib pure logic: data generator, optimizer, PyFunc"
```

> **Checkpoint:** none needed in-workspace (no MLflow yet). Proceed to Task B.

---

### Task B: Shared config + setup notebooks

**Files:**
- Create: `notebooks/_config.py`
- Create: `notebooks/00_setup.py`

**Produces (notebook-global after `%run ./_config`):** `CATALOG`, `SCHEMA`, `EXPERIMENT_PATH`, `FORECASTER_MODEL = f"{CATALOG}.{SCHEMA}.price_forecaster"`, `OPTIMIZER_MODEL = f"{CATALOG}.{SCHEMA}.purchase_optimizer"`, `DATA_TABLE = f"{CATALOG}.{SCHEMA}.commodity_monthly"`, `SEED`, `R2_THRESHOLD`.

- [ ] **Step 1: Write `notebooks/_config.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Shared config
# MAGIC `%run` this from every lab notebook. Widgets + names + reproducibility
# MAGIC constants (single source of truth lives in `workshop_lib`).

# COMMAND ----------
import workshop_lib as wl

dbutils.widgets.text("catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "mlops_workshop", "Schema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

EXPERIMENT_PATH = "/Shared/mlops_workshop"
FORECASTER_MODEL = f"{CATALOG}.{SCHEMA}.price_forecaster"
OPTIMIZER_MODEL = f"{CATALOG}.{SCHEMA}.purchase_optimizer"
DATA_TABLE = f"{CATALOG}.{SCHEMA}.commodity_monthly"
SEED = wl.SEED
R2_THRESHOLD = wl.R2_THRESHOLD
print(f"catalog={CATALOG} schema={SCHEMA} forecaster={FORECASTER_MODEL}")
```

- [ ] **Step 2: Write `notebooks/00_setup.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Installs deps, points MLflow at Unity Catalog, creates the catalog/schema
# MAGIC and experiment. Run once before the labs.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow
mlflow.set_registry_uri("databricks-uc")
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
mlflow.set_experiment(EXPERIMENT_PATH)
print("Setup complete.")
```

- [ ] **Step 3: Commit**
```bash
git add notebooks/_config.py notebooks/00_setup.py
git commit -m "Add shared config and setup notebooks"
```

> **Checkpoint:** author imports the repo into a Databricks Repo, sets `catalog`/`schema` widgets, runs `00_setup`. Expect `Setup complete.`, catalog/schema/experiment created. (If `%pip install -r ../requirements.txt` relative path fails, swap to an explicit package list — note for docs.)

---

### Task C: `01_generate_data.py` (inline R²-band assert)

**Files:** Create `notebooks/01_generate_data.py`.
**Consumes:** `workshop_lib`, `_config` (`DATA_TABLE`, `SEED`). **Produces:** Delta table `DATA_TABLE` + CSV; printed data dictionary/sanity checks; inline assert R² ∈ band.

- [ ] **Step 1: Write `notebooks/01_generate_data.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Generate the dataset
# MAGIC One shared, reproducible, time-ordered synthetic commodity dataset. All
# MAGIC models read from this. We assert the signal is learnable-but-not-trivial.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import workshop_lib as wl
df = wl.generate_dataset(seed=SEED)        # 96 monthly rows, time-ordered
display(df.head(10))

# COMMAND ----------
# MAGIC %md ## Data dictionary + sanity checks
# COMMAND ----------
print(df.describe())
print("\nDriver correlation with target:")
print(df[wl.DRIVERS].corrwith(df["price_next_month"]).sort_values())

from sklearn.metrics import r2_score
cut = int(len(df) * 0.8)
naive_r2 = r2_score(df["price_next_month"].iloc[cut:], df["price"].iloc[cut:])
model_r2 = wl.quick_fit_r2(df)
print(f"naive R2={naive_r2:.3f}  quick-model R2={model_r2:.3f}")

# COMMAND ----------
# Inline gate: fail loudly if the demo signal is mis-tuned.
lo, hi = wl.R2_BAND
assert lo <= model_r2 <= hi, f"R2 {model_r2:.3f} outside band {wl.R2_BAND}"

# COMMAND ----------
# Persist as Delta (governed) + CSV (portable).
spark.createDataFrame(df).write.mode("overwrite").saveAsTable(DATA_TABLE)
df.to_csv("/dbfs/tmp/commodity_monthly.csv", index=False)
print(f"Wrote {DATA_TABLE} and CSV.")
```

- [ ] **Step 2: Commit**
```bash
git add notebooks/01_generate_data.py
git commit -m "Add data-generation notebook with inline R2-band assert"
```

> **Checkpoint:** author runs `01`. Expect head/describe/correlations, `naive R2 < quick-model R2`, the assert passing, `SELECT count(*) FROM <DATA_TABLE>` = 96.

---

### Task D: `02_sklearn_baseline.py` (validation gate)

**Files:** Create `notebooks/02_sklearn_baseline.py`.
**Consumes:** `DATA_TABLE`, `FORECASTER_MODEL`, `R2_THRESHOLD`, `SEED`, `workshop_lib.DRIVERS`. **Produces:** UC model `FORECASTER_MODEL` + alias `@champion` iff test R² ≥ `R2_THRESHOLD`.

- [ ] **Step 1: Write `notebooks/02_sklearn_baseline.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · sklearn standard path
# MAGIC Train → track → evaluate against a fixed bar → register and promote to
# MAGIC `@champion` only if it passes. The everyday MLOps loop.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, numpy as np, workshop_lib as wl
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

df = spark.table(DATA_TABLE).toPandas().sort_values("month")
feats = wl.DRIVERS + ["price"]
cut = int(len(df) * 0.8)                     # time-respecting split, no shuffle
train, test = df.iloc[:cut], df.iloc[cut:]

# COMMAND ----------
with mlflow.start_run(run_name="sklearn_gbr") as run:
    model = GradientBoostingRegressor(random_state=SEED)
    model.fit(train[feats], train["price_next_month"])
    preds = model.predict(test[feats])
    rmse = float(np.sqrt(mean_squared_error(test["price_next_month"], preds)))
    mae = float(mean_absolute_error(test["price_next_month"], preds))
    r2 = float(r2_score(test["price_next_month"], preds))
    mlflow.log_params({"model_type": "GradientBoostingRegressor", "seed": SEED})
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.sklearn.log_model(
        model, name="model",
        input_example=test[feats].head(3),
        signature=mlflow.models.infer_signature(test[feats], preds),
    )
    print(f"r2={r2:.3f} rmse={rmse:.2f}")

# COMMAND ----------
# Validation gate: register + promote only if the model clears the bar.
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate (r2={r2:.3f} >= {R2_THRESHOLD}). Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f} < {R2_THRESHOLD}). Not promoted.")
```

- [ ] **Step 2: Commit**
```bash
git add notebooks/02_sklearn_baseline.py
git commit -m "Add sklearn notebook with deterministic validation gate"
```

> **Checkpoint:** author runs `02`. Expect a logged run, `PASSED gate`, `FORECASTER_MODEL` in UC, `@champion` resolves via `client.get_model_version_by_alias(FORECASTER_MODEL,"champion")`.

---

### Task E: `04_pyomo_pyfunc.py` (headline, inline feasibility assert)

**Files:** Create `notebooks/04_pyomo_pyfunc.py`.
**Consumes:** `OPTIMIZER_MODEL`, `workshop_lib.PurchaseOptimizerModel`/`ECON_COLS`. **Produces:** UC PyFunc `OPTIMIZER_MODEL` + alias `@champion`; loaded by alias and called; inline feasibility assert.

- [ ] **Step 1: Write `notebooks/04_pyomo_pyfunc.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Pyomo optimizer as a custom PyFunc (the headline)
# MAGIC A non-trainable operations-research model, wrapped as MLflow PyFunc, lives
# MAGIC in the exact same registry/lifecycle as the ML predictor.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, pandas as pd, workshop_lib as wl
from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

example = pd.DataFrame([{
    "predicted_price": 210.0, "holding_cost": 1.0, "purchase_cost": 200.0,
    "demand": 100.0, "capacity": 200.0, "budget": 30000.0,
}])
sample_out = wl.PurchaseOptimizerModel().predict(None, example)

# Inline sanity assert: the solve must be feasible and meet demand within capacity.
row = sample_out.iloc[0]
assert row["status"] == "optimal", f"optimizer infeasible on example: {row.to_dict()}"
assert 100.0 - 1e-6 <= row["purchase_qty"] <= 200.0 + 1e-6, f"qty out of bounds: {row['purchase_qty']}"
print(sample_out)

# COMMAND ----------
signature = mlflow.models.infer_signature(example, sample_out)
client = MlflowClient()
with mlflow.start_run(run_name="pyomo_optimizer"):
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=wl.PurchaseOptimizerModel(),
        code_paths=["./workshop_lib.py"],          # ship the optimizer logic
        pip_requirements=["pyomo>=6.7", "highspy>=1.7", "pandas>=2.0"],
        input_example=example,
        signature=signature,
        registered_model_name=OPTIMIZER_MODEL,
    )
# Promote by alias so the whole workshop loads uniformly by @champion.
client.set_registered_model_alias(OPTIMIZER_MODEL, "champion", info.registered_model_version)

# COMMAND ----------
# Load BY ALIAS and call — proves it is governed like any model.
opt = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
print(opt.predict(example))
```

- [ ] **Step 2: Commit**
```bash
git add notebooks/04_pyomo_pyfunc.py
git commit -m "Add Pyomo PyFunc notebook with inline feasibility assert"
```

> **Checkpoint:** author runs `04`. Expect the inline assert to pass, `OPTIMIZER_MODEL` registered, `@champion` set, the reloaded model printing `purchase_qty≈100`, `status=optimal`.

---

### Task F: `05_end_to_end.py` (chain + @champion guard)

**Files:** Create `notebooks/05_end_to_end.py`.
**Consumes:** `FORECASTER_MODEL@champion`, `OPTIMIZER_MODEL@champion`, `DATA_TABLE`. **Produces:** end-to-end decision for the latest month; depends only on the sklearn lab for `@champion`.

- [ ] **Step 1: Write `notebooks/05_end_to_end.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · End-to-end chain
# MAGIC Load the registered predictor (`@champion`) and optimizer (`@champion`),
# MAGIC then run drivers → predicted price → purchase decision on the latest month.
# MAGIC A once-manual monthly run now lives in a governed lineage.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, pandas as pd, workshop_lib as wl
from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Fail clearly if the predictor was never promoted, instead of a raw registry error.
try:
    client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
except Exception:
    raise RuntimeError(
        f"No @champion alias on {FORECASTER_MODEL}. Run 02_sklearn_baseline and "
        f"confirm it passed the R2>={R2_THRESHOLD} gate before running this notebook."
    )

df = spark.table(DATA_TABLE).toPandas().sort_values("month")
latest = df.iloc[[-1]]

forecaster = mlflow.pyfunc.load_model(f"models:/{FORECASTER_MODEL}@champion")
optimizer = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")

# COMMAND ----------
predicted_price = float(forecaster.predict(latest[wl.DRIVERS + ["price"]])[0])
opt_input = latest[wl.ECON_COLS].copy()
opt_input.insert(0, "predicted_price", predicted_price)
decision = optimizer.predict(opt_input)
print(f"Predicted next-month price: {predicted_price:.2f}")
print(decision)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Governance & lineage
# MAGIC Open **Catalog Explorer → the Delta table → Lineage** to see table →
# MAGIC experiment → registered models. The ungoverned monthly manual run now has
# MAGIC reproducible inputs, tracked runs, registered models, and lineage.
```

- [ ] **Step 2: Commit**
```bash
git add notebooks/05_end_to_end.py
git commit -m "Add end-to-end chain notebook with @champion guard and lineage beat"
```

> **Checkpoint:** author runs `05` (after `02` and `04`). Expect a predicted price + one-row decision (`status=optimal`); confirm it runs without ever having run the PyTorch lab.

---

### Task G: `06_verify.py` (four checkpoints)

**Files:** Create `notebooks/06_verify.py`.
**Consumes:** all above. **Produces:** assertions for the four DoD checkpoints, alias-based.

- [ ] **Step 1: Write `notebooks/06_verify.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Verify (minimum bar)
# MAGIC The four definition-of-done checkpoints on the mandatory path. Assertions
# MAGIC are alias-based (re-runs bump version numbers).

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, workshop_lib as wl
from mlflow import MlflowClient
import pandas as pd
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 1) R2 in band
df = spark.table(DATA_TABLE).toPandas().sort_values("month")
r2 = wl.quick_fit_r2(df)
assert wl.R2_BAND[0] <= r2 <= wl.R2_BAND[1], f"R2 {r2:.3f} outside {wl.R2_BAND}"

# 2) forecaster registered
assert client.get_registered_model(FORECASTER_MODEL) is not None

# 3) @champion resolves (alias, never a literal version integer)
champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
assert champ is not None and champ.version is not None

# 4) chain returns a purchase decision
forecaster = mlflow.pyfunc.load_model(f"models:/{FORECASTER_MODEL}@champion")
optimizer = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
latest = df.iloc[[-1]]
pp = float(forecaster.predict(latest[wl.DRIVERS + ["price"]])[0])
oi = latest[wl.ECON_COLS].copy(); oi.insert(0, "predicted_price", pp)
dec = optimizer.predict(oi)
assert dec.loc[0, "status"] == "optimal" and pd.notna(dec.loc[0, "purchase_qty"])
print("ALL 4 CHECKPOINTS PASSED")
```

- [ ] **Step 2: Commit**
```bash
git add notebooks/06_verify.py
git commit -m "Add 4-checkpoint verification notebook"
```

> **Checkpoint:** author runs `06` after `01→02→04→05`. Expect `ALL 4 CHECKPOINTS PASSED`.

---

### Task H: `97_optional_pytorch.py` (optional, non-dependency)

**Files:** Create `notebooks/97_optional_pytorch.py`.
**Consumes:** `DATA_TABLE`, `FORECASTER_MODEL`, `R2_THRESHOLD`, `SEED`, `workshop_lib.DRIVERS`. **Produces:** a new version of `FORECASTER_MODEL` + `@champion` re-pointed iff R² ≥ threshold. Not a dependency of anything.

- [ ] **Step 1: Write `notebooks/97_optional_pytorch.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 97 · Optional — PyTorch (same lifecycle, different flavor)
# MAGIC The SAME data/split through a small MLP, reusing the SAME validation gate.
# MAGIC Optional: the end-to-end chain does not depend on this notebook.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, numpy as np, torch, torch.nn as nn, workshop_lib as wl
from mlflow import MlflowClient
from sklearn.metrics import r2_score, mean_squared_error
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
torch.manual_seed(SEED)

df = spark.table(DATA_TABLE).toPandas().sort_values("month")
feats = wl.DRIVERS + ["price"]
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
mu, sd = train[feats].mean(), train[feats].std(ddof=0)

def to_t(frame):
    return torch.tensor(((frame[feats] - mu) / sd).to_numpy(), dtype=torch.float32)

Xtr = to_t(train); ytr = torch.tensor(train["price_next_month"].to_numpy(), dtype=torch.float32).view(-1, 1)
Xte = to_t(test)

# COMMAND ----------
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x)

with mlflow.start_run(run_name="pytorch_mlp"):
    net = MLP(len(feats)); opt = torch.optim.Adam(net.parameters(), lr=0.01); loss_fn = nn.MSELoss()
    for _ in range(400):
        opt.zero_grad(); loss = loss_fn(net(Xtr), ytr); loss.backward(); opt.step()
    preds = net(Xte).detach().numpy().ravel()
    r2 = float(r2_score(test["price_next_month"], preds))
    rmse = float(np.sqrt(mean_squared_error(test["price_next_month"], preds)))
    mlflow.log_params({"model_type": "torch_mlp", "epochs": 400, "seed": SEED})
    mlflow.log_metrics({"r2": r2, "rmse": rmse})
    info = mlflow.pytorch.log_model(net, name="model")
    print(f"r2={r2:.3f} rmse={rmse:.2f}")

# COMMAND ----------
# Same gate as the sklearn lab.
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(info.model_uri, FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate. Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f}). Not promoted.")
```

- [ ] **Step 2: Commit**
```bash
git add notebooks/97_optional_pytorch.py
git commit -m "Add optional PyTorch notebook reusing the validation gate"
```

> **Checkpoint:** author runs `97` independently. Expect a `pytorch_mlp` run, gate PASS/FAIL; if PASS, a new `FORECASTER_MODEL` version. Confirm `05`/`06` still pass regardless (they load whatever `@champion` points to).

---

### Task I: `99_optional_serving.py` + notebooks README

**Files:** Create `notebooks/99_optional_serving.py`, `notebooks/README.md`.
**Consumes:** `FORECASTER_MODEL`, `OPTIMIZER_MODEL`. **Produces:** a served forecaster endpoint (+ optimizer note); a README of run order.

- [ ] **Step 1: Write `notebooks/99_optional_serving.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · Optional — Model Serving
# MAGIC Deploy a registered model to a serving endpoint and query it. The Pyomo
# MAGIC endpoint works because `highspy` is pip-installable inside the container.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
from mlflow.deployments import get_deploy_client
from mlflow import MlflowClient
client = MlflowClient()
deploy = get_deploy_client("databricks")

champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
endpoint = "mlops-workshop-forecaster"
deploy.create_endpoint(
    name=endpoint,
    config={"served_entities": [{
        "entity_name": FORECASTER_MODEL, "entity_version": champ.version,
        "workload_size": "Small", "scale_to_zero_enabled": True,
    }]},
)
print(f"Creating endpoint {endpoint} for {FORECASTER_MODEL} v{champ.version}.")

# COMMAND ----------
# MAGIC %md
# MAGIC When Ready, query it (drivers + price columns) via the Serving UI or REST.
# MAGIC To serve the Pyomo optimizer, create a second endpoint for `OPTIMIZER_MODEL`
# MAGIC — its `pip_requirements` already include `highspy`/`pyomo`.
```

- [ ] **Step 2: Write `notebooks/README.md`**

```markdown
# Workshop notebooks

Run order (mandatory path): `00_setup` → `01_generate_data` → `02_sklearn_baseline`
→ `04_pyomo_pyfunc` → `05_end_to_end` → `06_verify`.

Optional: `97_optional_pytorch`, `99_optional_serving`, `98_optional_cleanup` (teardown).

`workshop_lib.py` holds the pure, reusable logic (data generation, the Pyomo
optimizer, the PyFunc wrapper); the notebooks import it rather than duplicating
logic. Verification is inline asserts in the notebooks plus `06_verify`.

Set the `catalog` and `schema` widgets in `_config` to your own Unity Catalog
target. All model loads use the `@champion` alias, never version numbers.
```

- [ ] **Step 3: Commit**
```bash
git add notebooks/99_optional_serving.py notebooks/README.md
git commit -m "Add optional serving notebook and notebooks README"
```

> **Checkpoint:** author runs `99` (after the forecaster is registered). Expect endpoint creation to start; once Ready, a query returns a prediction. (Provisioning takes several minutes — fine for an optional lab.)

---

### Task J: `98_optional_cleanup.py` (optional teardown)

**Files:** Create `notebooks/98_optional_cleanup.py`.
**Consumes:** `FORECASTER_MODEL`, `OPTIMIZER_MODEL`, `EXPERIMENT_PATH`, `DATA_TABLE`, `CATALOG`, `SCHEMA`. **Produces:** removes all resources the workshop created. Destructive; guarded by a `confirm` widget (inline fail-loud assert). Idempotent (`IF EXISTS` / try-except).

- [ ] **Step 1: Write `notebooks/98_optional_cleanup.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 98 · Optional — Cleanup (teardown)
# MAGIC Removes everything the workshop created: serving endpoint, registered
# MAGIC models, the experiment, the Delta table (and optionally the schema).
# MAGIC **Destructive.** Set the `confirm` widget to `yes` to proceed.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
dbutils.widgets.dropdown("confirm", "no", ["no", "yes"], "Confirm teardown")
dbutils.widgets.dropdown("drop_schema", "no", ["no", "yes"], "Also drop schema (CASCADE)")
assert dbutils.widgets.get("confirm") == "yes", \
    "Set the 'confirm' widget to 'yes' to delete workshop resources."

# COMMAND ----------
import mlflow
from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 1) Serving endpoint (the optional serving lab may have created it).
try:
    get_deploy_client("databricks").delete_endpoint("mlops-workshop-forecaster")
    print("Deleted serving endpoint.")
except Exception as e:
    print(f"No serving endpoint to delete ({e}).")

# 2) Registered models.
for name in (FORECASTER_MODEL, OPTIMIZER_MODEL):
    try:
        client.delete_registered_model(name)
        print(f"Deleted registered model {name}.")
    except Exception as e:
        print(f"No model {name} ({e}).")

# 3) Experiment.
try:
    exp = client.get_experiment_by_name(EXPERIMENT_PATH)
    if exp:
        mlflow.delete_experiment(exp.experiment_id)
        print(f"Deleted experiment {EXPERIMENT_PATH}.")
except Exception as e:
    print(f"No experiment to delete ({e}).")

# 4) Data table + CSV.
spark.sql(f"DROP TABLE IF EXISTS {DATA_TABLE}")
try:
    dbutils.fs.rm("dbfs:/tmp/commodity_monthly.csv")
except Exception:
    pass
print(f"Dropped {DATA_TABLE} and CSV.")

# COMMAND ----------
# Optional: drop the whole schema (removes anything left behind). Off by default
# because the catalog/schema may be shared.
if dbutils.widgets.get("drop_schema") == "yes":
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
    print(f"Dropped schema {CATALOG}.{SCHEMA}.")
print("Cleanup complete.")
```

- [ ] **Step 2: Commit**
```bash
git add notebooks/98_optional_cleanup.py
git commit -m "Add optional cleanup/teardown notebook"
```

> **Checkpoint:** author runs `98` with `confirm=yes`. Expect each resource deleted or a clean "nothing to delete" message; re-running is safe (idempotent). The `drop_schema` toggle stays `no` unless the schema is dedicated to the workshop.

---

## Coverage check (vs. spec)
- Agnostic, synthetic, fixed seed → Global Constraints + Task A. ✓
- Two-act narrative (sklearn standard, Pyomo headline) → Tasks D, E. ✓
- Optional PyTorch, non-dependency → Task H; Task F depends only on `@champion`. ✓
- Deterministic single-model gate (R²≥0.6) → Tasks D, H. ✓
- UC widget-parameterized registry, alias-first (no `/latest`, no version ints) → Tasks B, D, E, F, G. ✓
- Data: 96 months, time-ordered, 10 drivers, causal target, inline R²-band assert, `trend_up` derived-only, econ cols → Tasks A, C. ✓
- Pyomo HiGHS PyFunc, separate model, inline feasibility assert, load & call → Tasks A, E. ✓
- End-to-end chain + lineage beat → Task F. ✓
- Four-checkpoint verification, alias-based → Task G. ✓
- Optional serving + solver-in-container note → Task I. ✓
- Optional cleanup/teardown (guarded, idempotent) → Task J. ✓
- `.py` source format, `_config` via `%run`, notebooks import `workshop_lib` → Tasks A–I. ✓
- No test suite (inline asserts only) → Global Constraints; no `tests/` created. ✓
