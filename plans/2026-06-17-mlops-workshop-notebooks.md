# MLOps Workshop — Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Databricks `.py` source notebooks for an MLOps workshop where one shared synthetic dataset flows through a standard ML predictor (sklearn) and an unusual custom OR model (Pyomo), all governed by one MLflow lifecycle on Unity Catalog.

**Architecture:** Pure, reusable logic (synthetic data generation, the Pyomo purchase optimizer, and the PyFunc wrapper) lives in a small importable module `notebooks/workshop_lib.py` that is unit-tested locally with pytest — no Databricks required. The Databricks notebooks are thin orchestration layers that import that logic and add the MLflow/Unity-Catalog/registry/serving glue, which is verified in-workspace via a 4-checkpoint harness. `_config.py` centralizes widgets and the reproducibility constants.

**Tech Stack:** Python 3.10+, pandas, NumPy, scikit-learn, Pyomo + HiGHS (`highspy`, `appsi_highs`), PyTorch (optional lab), MLflow (UC registry), Databricks notebooks (`.py` source format), pytest (local verification).

## Global Constraints

- **Customer-agnostic:** no real customer/business-unit/model names anywhere (code, docs, comments, commits). Use fictional **"AnyCompany"** only if a name is needed. Synthetic data only.
- **Notebook format:** Databricks source `.py` — first line `# Databricks notebook source`, cells separated by `# COMMAND ----------`, markdown cells via `# MAGIC %md`.
- **Registry:** Unity Catalog (`mlflow.set_registry_uri("databricks-uc")`), names `{catalog}.{schema}.<model>`, widget-parameterized — no hardcoded workspace.
- **Alias-first:** all cross-notebook model references use the `@champion` alias — **never literal version integers** (re-runs bump versions).
- **Solver:** HiGHS via `highspy` through Pyomo `appsi_highs` (pure pip, no system binary).
- **Reproducibility:** single named `SEED` constant; achievable test R² must land in **[0.6, 0.85]**; promotion gate threshold `R2_THRESHOLD = 0.6`.
- **Regression-only:** `trend_up` is a derived illustration column, never modeled.
- **Mandatory notebook path:** `01 → 02 → 04 → 05`. PyTorch (`97_`) and serving (`99_`) are optional and must not be dependencies of the finale.
- **Do not** create or modify the MkDocs site files (`docs/`, `mkdocs.yml`, `.github/`) — that is a separate plan.

---

### Task 1: Repo scaffolding, dependency pinning, local test harness

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `notebooks/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: a working local Python environment where `pytest` runs, and pinned runtime deps the notebooks rely on.

- [ ] **Step 1: Create `requirements.txt` (notebook runtime deps, pinned floors)**

```
# Workshop notebook runtime dependencies (Databricks ML runtime compatible)
mlflow>=2.15,<3
scikit-learn>=1.3
pandas>=2.0
numpy>=1.26
pyomo>=6.7
highspy>=1.7
torch>=2.2          # optional PyTorch lab only
```

- [ ] **Step 2: Create `requirements-dev.txt` (local test deps)**

```
-r requirements.txt
pytest>=8.0
```

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v
```

- [ ] **Step 4: Create placeholder files**

```bash
mkdir -p notebooks tests
touch notebooks/.gitkeep tests/__init__.py
printf '# Make notebooks/ importable in local tests\nimport sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "notebooks"))\n' > tests/conftest.py
```

- [ ] **Step 5: Create the local venv and install dev deps**

Run:
```bash
python3 -m venv .venv && .venv/bin/pip install --quiet -r requirements-dev.txt && .venv/bin/python -c "import pyomo, highspy, sklearn, pandas; print('deps ok')"
```
Expected: `deps ok` (confirms `highspy`/`pyomo` install via pip with no system binary). Add `.venv/` to `.gitignore` if not already ignored.

- [ ] **Step 6: Verify pytest runs (no tests yet)**

Run: `.venv/bin/pytest`
Expected: exit code 5 / "no tests ran" — confirms the harness is wired.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt requirements-dev.txt pytest.ini tests/ notebooks/.gitkeep .gitignore
git commit -m "Scaffold notebook deps and local pytest harness"
```

---

### Task 2: Synthetic data generator + reproducibility constants (TDD, local)

**Files:**
- Create: `notebooks/workshop_lib.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SEED: int` and `R2_THRESHOLD: float = 0.6`, `R2_BAND = (0.6, 0.85)` constants.
  - `DRIVERS: list[str]` — the 10 driver column names.
  - `ECON_COLS: list[str]` — `["holding_cost", "purchase_cost", "demand", "capacity", "budget"]`.
  - `generate_dataset(n_months: int = 96, seed: int = SEED) -> pandas.DataFrame` — time-ordered monthly frame containing `month` (int 0..n-1, monotonic), the 10 `DRIVERS`, `price` (current), `price_next_month` (target), `trend_up` (bool), and the 5 `ECON_COLS`.
  - `quick_fit_r2(df: pandas.DataFrame) -> float` — fits a small GradientBoostingRegressor on `DRIVERS + ["price"]` → `price_next_month` with a time-ordered 80/20 split and returns test R².

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data.py
import numpy as np
import pandas as pd
import workshop_lib as wl


def test_constants_present():
    assert isinstance(wl.SEED, int)
    assert wl.R2_THRESHOLD == 0.6
    assert wl.R2_BAND == (0.6, 0.85)
    assert len(wl.DRIVERS) == 10
    assert wl.ECON_COLS == ["holding_cost", "purchase_cost", "demand", "capacity", "budget"]


def test_schema_and_ordering():
    df = wl.generate_dataset(n_months=96)
    assert len(df) == 96
    for col in ["month", "price", "price_next_month", "trend_up", *wl.DRIVERS, *wl.ECON_COLS]:
        assert col in df.columns, f"missing {col}"
    # time-ordered, never shuffled
    assert df["month"].is_monotonic_increasing
    assert list(df["month"]) == list(range(96))
    assert df["trend_up"].dtype == bool


def test_trend_up_is_derived():
    df = wl.generate_dataset(n_months=96)
    expected = df["price_next_month"] > df["price"]
    assert (df["trend_up"] == expected).all()


def test_reproducible_with_seed():
    a = wl.generate_dataset(n_months=96, seed=123)
    b = wl.generate_dataset(n_months=96, seed=123)
    pd.testing.assert_frame_equal(a, b)
    c = wl.generate_dataset(n_months=96, seed=999)
    assert not a["price_next_month"].equals(c["price_next_month"])


def test_signal_lands_in_band():
    df = wl.generate_dataset(n_months=96)
    r2 = wl.quick_fit_r2(df)
    lo, hi = wl.R2_BAND
    assert lo <= r2 <= hi, f"R2={r2:.3f} outside band {wl.R2_BAND}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_data.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'workshop_lib'` (or attribute errors).

- [ ] **Step 3: Implement `workshop_lib.py` (data section)**

```python
"""Reusable, framework-agnostic logic for the MLOps workshop.

Imported by both the Databricks notebooks and the local pytest suite, so it must
not depend on Databricks, MLflow, or dbutils. Pure pandas/NumPy/Pyomo only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Reproducibility & gate constants (single source of truth) ---------------
SEED: int = 42
R2_THRESHOLD: float = 0.6          # promotion gate: register/@champion only if test R2 >= this
R2_BAND: tuple[float, float] = (0.6, 0.85)  # data generator must keep achievable R2 in this band

DRIVERS: list[str] = [
    "demand_index", "input_cost_index", "fx_rate", "inventory_level",
    "industrial_output", "energy_cost", "scrap_supply", "export_demand",
    "seasonality", "competitor_price",
]
ECON_COLS: list[str] = ["holding_cost", "purchase_cost", "demand", "capacity", "budget"]


def generate_dataset(n_months: int = 96, seed: int = SEED) -> pd.DataFrame:
    """Time-ordered monthly synthetic commodity dataset with a causal target.

    The target ``price_next_month`` is a weighted function of the drivers plus the
    lagged price plus noise, so an ML model can beat a naive baseline but not
    perfectly (tune toward R2 in R2_BAND). Rows are time-ordered and never shuffled.
    """
    rng = np.random.default_rng(seed)
    n = n_months

    # Drivers: plausible, mildly autocorrelated commodity-market signals.
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

    # Standardize drivers for stable weighting, then build a causal current price.
    z = (df[DRIVERS] - df[DRIVERS].mean()) / df[DRIVERS].std(ddof=0)
    weights = np.array([8, 6, 5, -3, 4, 5, -4, 3, 2, 7], dtype=float)
    base = 200.0 + z.to_numpy() @ weights
    df["price"] = base + rng.normal(0, 6, n)

    # Target: next month's price = f(drivers, lagged price) + noise. Last row uses
    # current price as a stand-in (it is the "to-be-forecast" latest month).
    signal = 0.6 * df["price"].to_numpy() + 0.4 * base
    noise = rng.normal(0, 14, n)             # tuned so quick_fit_r2 lands in band
    price_next = signal + noise
    price_next[:-1] = price_next[1:]         # shift so row t predicts t+1
    df["price_next_month"] = price_next
    df["trend_up"] = df["price_next_month"] > df["price"]

    # Decision-economics columns (ignored by predictors; consumed by the optimizer).
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
```

- [ ] **Step 4: Run tests; iterate noise/weights until R² lands in band**

Run: `.venv/bin/pytest tests/test_data.py`
Expected: PASS. If `test_signal_lands_in_band` fails, adjust the `noise` std (larger → lower R²) until test R² ∈ [0.6, 0.85]. Re-run until green.

- [ ] **Step 5: Commit**

```bash
git add notebooks/workshop_lib.py tests/test_data.py
git commit -m "Add synthetic data generator with causal target and R2-band test"
```

---

### Task 3: Pyomo purchase optimizer (TDD, local)

**Files:**
- Modify: `notebooks/workshop_lib.py`
- Test: `tests/test_optimizer.py`

**Interfaces:**
- Consumes: nothing (pure Pyomo).
- Produces:
  - `solve_purchase(predicted_price: float, holding_cost: float, purchase_cost: float, demand: float, capacity: float, budget: float) -> dict` returning `{"purchase_qty": float, "total_cost": float, "status": str}`. Minimizes `purchase_cost*q + holding_cost*max(q-demand,0)` (modeled with a leftover var) s.t. `q >= demand`, `q <= capacity`, `purchase_cost*q <= budget`. `status` is `"optimal"` or `"infeasible"`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_optimizer.py
import pytest
import workshop_lib as wl


def test_meets_demand_when_cheap_to_hold():
    r = wl.solve_purchase(predicted_price=100, holding_cost=0.1,
                          purchase_cost=10, demand=100, capacity=200, budget=5000)
    assert r["status"] == "optimal"
    assert r["purchase_qty"] >= 100 - 1e-6          # demand met
    assert r["purchase_qty"] <= 200 + 1e-6          # capacity respected


def test_buys_exactly_demand_when_holding_is_expensive():
    r = wl.solve_purchase(predicted_price=100, holding_cost=5.0,
                          purchase_cost=10, demand=100, capacity=200, budget=5000)
    assert r["status"] == "optimal"
    assert r["purchase_qty"] == pytest.approx(100, abs=1e-3)  # no incentive to over-buy


def test_budget_can_make_it_infeasible():
    # demand=100 needs >=100 units at cost 10 = 1000, but budget only 500
    r = wl.solve_purchase(predicted_price=100, holding_cost=0.1,
                          purchase_cost=10, demand=100, capacity=200, budget=500)
    assert r["status"] == "infeasible"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_optimizer.py`
Expected: FAIL — `AttributeError: module 'workshop_lib' has no attribute 'solve_purchase'`.

- [ ] **Step 3: Implement `solve_purchase` in `workshop_lib.py`**

```python
def solve_purchase(predicted_price: float, holding_cost: float, purchase_cost: float,
                   demand: float, capacity: float, budget: float) -> dict:
    """Single-period purchase decision: minimize purchase + holding cost subject to
    meeting demand within capacity and budget. Solved with HiGHS via Pyomo appsi."""
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers.highs import Highs

    m = pyo.ConcreteModel()
    m.q = pyo.Var(domain=pyo.NonNegativeReals)          # purchase quantity
    m.leftover = pyo.Var(domain=pyo.NonNegativeReals)   # units held after demand

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
    return {
        "purchase_qty": float(pyo.value(m.q)),
        "total_cost": float(pyo.value(m.obj)),
        "status": "optimal",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_optimizer.py`
Expected: PASS. (If the `appsi` import path differs in the installed Pyomo version, fall back to `pyo.SolverFactory("appsi_highs")`; confirm the termination-condition string matches the `"optimal"`/`"infeasible"` check.)

- [ ] **Step 5: Commit**

```bash
git add notebooks/workshop_lib.py tests/test_optimizer.py
git commit -m "Add Pyomo HiGHS purchase optimizer with feasibility tests"
```

---

### Task 4: PyFunc wrapper for the optimizer (TDD, local)

**Files:**
- Modify: `notebooks/workshop_lib.py`
- Test: `tests/test_pyfunc.py`

**Interfaces:**
- Consumes: `solve_purchase`.
- Produces: `PurchaseOptimizerModel` (subclass of `mlflow.pyfunc.PythonModel`) whose `predict(self, context, model_input: pandas.DataFrame) -> pandas.DataFrame` reads columns `["predicted_price", *ECON_COLS]` row-wise, calls `solve_purchase`, and returns a frame with columns `["purchase_qty", "total_cost", "status"]`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_pyfunc.py
import pandas as pd
import workshop_lib as wl


def test_pyfunc_predict_returns_decisions():
    model = wl.PurchaseOptimizerModel()
    X = pd.DataFrame([{
        "predicted_price": 100, "holding_cost": 5.0, "purchase_cost": 10,
        "demand": 100, "capacity": 200, "budget": 5000,
    }])
    out = model.predict(None, X)
    assert list(out.columns) == ["purchase_qty", "total_cost", "status"]
    assert out.loc[0, "status"] == "optimal"
    assert abs(out.loc[0, "purchase_qty"] - 100) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pyfunc.py`
Expected: FAIL — no `PurchaseOptimizerModel`.

- [ ] **Step 3: Implement `PurchaseOptimizerModel` in `workshop_lib.py`**

```python
import mlflow.pyfunc  # safe: mlflow is a notebook dep; only imported where used


class PurchaseOptimizerModel(mlflow.pyfunc.PythonModel):
    """Wraps the Pyomo purchase optimizer as an MLflow PyFunc so a non-trainable
    OR model lives in the same registry/lifecycle as the ML predictor."""

    def predict(self, context, model_input):
        import pandas as pd
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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_pyfunc.py`
Expected: PASS.

- [ ] **Step 5: Run the full local suite + commit**

Run: `.venv/bin/pytest`
Expected: all tests PASS.
```bash
git add notebooks/workshop_lib.py tests/test_pyfunc.py
git commit -m "Wrap purchase optimizer as MLflow PyFunc with predict test"
```

---

### Task 5: Shared config + setup notebooks

**Files:**
- Create: `notebooks/_config.py`
- Create: `notebooks/00_setup.py`

**Interfaces:**
- Consumes: `workshop_lib` constants (`SEED`, `R2_THRESHOLD`, `DRIVERS`, `ECON_COLS`).
- Produces (as notebook-global names after `%run ./_config`): `CATALOG`, `SCHEMA`, `EXPERIMENT_PATH`, `FORECASTER_MODEL = f"{CATALOG}.{SCHEMA}.price_forecaster"`, `OPTIMIZER_MODEL = f"{CATALOG}.{SCHEMA}.purchase_optimizer"`, `DATA_TABLE = f"{CATALOG}.{SCHEMA}.commodity_monthly"`, `SEED`, `R2_THRESHOLD`.

> **Verification note:** Tasks 5–11 produce Databricks notebooks that run in a workspace; they cannot be executed locally. Each step gives the exact cell content and the expected in-workspace result. Authoritative verification is Task 11's harness, run manually in the workspace.

- [ ] **Step 1: Write `notebooks/_config.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Shared config
# MAGIC `%run` this from every lab notebook. Defines widgets, names, and the
# MAGIC reproducibility constants (single source of truth in `workshop_lib`).

# COMMAND ----------
import workshop_lib as wl

dbutils.widgets.text("catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "mlops_workshop", "Schema")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

EXPERIMENT_PATH = f"/Shared/mlops_workshop"
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
# MAGIC and experiment. Run this once before the labs.

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

- [ ] **Step 3: In-workspace verification**

Import the repo into a Databricks Repo, set the `catalog`/`schema` widgets, Run All on `00_setup.py`.
Expected: `Setup complete.` printed; the catalog/schema exist; the experiment appears under `/Shared/mlops_workshop`. (If the `%pip install -r ../requirements.txt` relative path fails in the workspace, replace with an explicit package list — note this for the docs.)

- [ ] **Step 4: Commit**

```bash
git add notebooks/_config.py notebooks/00_setup.py
git commit -m "Add shared config and setup notebooks"
```

---

### Task 6: Data-generation notebook

**Files:**
- Create: `notebooks/01_generate_data.py`

**Interfaces:**
- Consumes: `workshop_lib.generate_dataset`, `quick_fit_r2`, `R2_BAND`, `DRIVERS`; `_config` names `DATA_TABLE`, `SEED`.
- Produces: a Delta table `DATA_TABLE` and a CSV under a workspace volume/path; printed data dictionary + sanity checks; an assertion that achievable R² ∈ `R2_BAND`.

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

# Naive baseline: predict last month's price.
naive_pred = df["price"].iloc[int(len(df)*0.8):]
from sklearn.metrics import r2_score
naive_r2 = r2_score(df["price_next_month"].iloc[int(len(df)*0.8):], naive_pred)
model_r2 = wl.quick_fit_r2(df)
print(f"naive R2={naive_r2:.3f}  quick-model R2={model_r2:.3f}")

# COMMAND ----------
# Fail loudly if the demo signal is mis-tuned.
lo, hi = wl.R2_BAND
assert lo <= model_r2 <= hi, f"R2 {model_r2:.3f} outside band {wl.R2_BAND}"

# COMMAND ----------
# Persist as Delta (governed) + CSV (portable).
sdf = spark.createDataFrame(df)
sdf.write.mode("overwrite").saveAsTable(DATA_TABLE)
df.to_csv("/dbfs/tmp/commodity_monthly.csv", index=False)
print(f"Wrote {DATA_TABLE} and CSV.")
```

- [ ] **Step 2: In-workspace verification**

Run All. Expected: head/describe/correlations print; `naive R2 < quick-model R2`; the band assertion passes; `DATA_TABLE` is queryable (`SELECT count(*) FROM <DATA_TABLE>` → 96).

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_generate_data.py
git commit -m "Add data-generation notebook with sanity checks and R2-band gate"
```

---

### Task 7: sklearn standard-path notebook with validation gate

**Files:**
- Create: `notebooks/02_sklearn_baseline.py`

**Interfaces:**
- Consumes: `DATA_TABLE`, `FORECASTER_MODEL`, `R2_THRESHOLD`, `SEED`, `workshop_lib.DRIVERS`.
- Produces: a registered UC model `FORECASTER_MODEL` with alias `@champion` set **iff** test R² ≥ `R2_THRESHOLD`.

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
import mlflow, workshop_lib as wl
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

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
        model, "model",
        input_example=test[feats].head(3),
        signature=mlflow.models.infer_signature(test[feats], preds),
    )
    print(f"r2={r2:.3f} rmse={rmse:.2f}")

# COMMAND ----------
# Validation gate: register + promote only if the model clears the bar.
from mlflow import MlflowClient
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate (r2={r2:.3f} >= {R2_THRESHOLD}). Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f} < {R2_THRESHOLD}). Not promoted.")
```

- [ ] **Step 2: In-workspace verification**

Run All. Expected: a run logged under the experiment with `r2`/`rmse`/`mae`; `PASSED gate` printed; `FORECASTER_MODEL` exists in UC and `@champion` resolves (`client.get_model_version_by_alias(FORECASTER_MODEL, "champion")`).

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_sklearn_baseline.py
git commit -m "Add sklearn notebook with deterministic validation gate"
```

---

### Task 8: Pyomo PyFunc notebook (headline)

**Files:**
- Create: `notebooks/04_pyomo_pyfunc.py`

**Interfaces:**
- Consumes: `OPTIMIZER_MODEL`, `workshop_lib.PurchaseOptimizerModel`, `ECON_COLS`.
- Produces: a registered UC PyFunc model `OPTIMIZER_MODEL` (no alias gate — it is not trained/scored), loaded back and called.

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
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

example = pd.DataFrame([{
    "predicted_price": 210.0, "holding_cost": 1.0, "purchase_cost": 200.0,
    "demand": 100.0, "capacity": 200.0, "budget": 30000.0,
}])
signature = mlflow.models.infer_signature(
    example, wl.PurchaseOptimizerModel().predict(None, example))

# COMMAND ----------
with mlflow.start_run(run_name="pyomo_optimizer"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wl.PurchaseOptimizerModel(),
        code_paths=["./workshop_lib.py"],          # ship the optimizer logic
        pip_requirements=["pyomo>=6.7", "highspy>=1.7", "pandas>=2.0"],
        input_example=example,
        signature=signature,
        registered_model_name=OPTIMIZER_MODEL,
    )

# COMMAND ----------
# Load from the registry and call it — proves it is governed like any model.
opt = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion") \
      if False else mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}/latest")
print(opt.predict(example))
```

> **Interface note:** the optimizer has no quality gate, so it has no `@champion` alias by default. Task 10's end-to-end loads it by `models:/{OPTIMIZER_MODEL}/latest` (or set an explicit `production` alias here if preferred — keep it alias-based, not a version integer). Pick one and keep Task 10 consistent.

- [ ] **Step 2: In-workspace verification**

Run All. Expected: `OPTIMIZER_MODEL` registered; the loaded model prints a one-row frame with `purchase_qty` ≈ 100 and `status="optimal"`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_pyomo_pyfunc.py
git commit -m "Add Pyomo PyFunc notebook, registered and loaded from UC"
```

---

### Task 9: End-to-end chain notebook

**Files:**
- Create: `notebooks/05_end_to_end.py`

**Interfaces:**
- Consumes: `FORECASTER_MODEL@champion`, `OPTIMIZER_MODEL`, `DATA_TABLE`, `workshop_lib.DRIVERS`, `ECON_COLS`.
- Produces: a printed end-to-end decision for the latest month; lineage narrative. Depends only on the sklearn lab for `@champion`.

- [ ] **Step 1: Write `notebooks/05_end_to_end.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · End-to-end chain
# MAGIC Load the registered predictor (`@champion`) and the optimizer, then run
# MAGIC drivers → predicted price → purchase decision on the latest month. This is
# MAGIC the payoff: a once-manual monthly run now lives in a governed lineage.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, pandas as pd, workshop_lib as wl
mlflow.set_registry_uri("databricks-uc")

df = spark.table(DATA_TABLE).toPandas().sort_values("month")
latest = df.iloc[[-1]]
feats = wl.DRIVERS + ["price"]

forecaster = mlflow.pyfunc.load_model(f"models:/{FORECASTER_MODEL}@champion")
optimizer = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}/latest")

# COMMAND ----------
predicted_price = float(forecaster.predict(latest[feats])[0])
opt_input = latest[wl.ECON_COLS].copy()
opt_input.insert(0, "predicted_price", predicted_price)
decision = optimizer.predict(opt_input)

print(f"Predicted next-month price: {predicted_price:.2f}")
print(decision)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Governance & lineage
# MAGIC Open **Catalog Explorer → the Delta table → Lineage** to see the table →
# MAGIC experiment → registered models graph. The ungoverned monthly manual run now
# MAGIC has reproducible inputs, tracked runs, registered models, and lineage.
```

- [ ] **Step 2: In-workspace verification**

Run All (after Tasks 7 and 8). Expected: prints a predicted price and a one-row decision frame with `status="optimal"` and a numeric `purchase_qty`. Confirm it runs without ever having run the PyTorch lab.

- [ ] **Step 3: Commit**

```bash
git add notebooks/05_end_to_end.py
git commit -m "Add end-to-end chain notebook with lineage beat"
```

---

### Task 10: Verification harness notebook (4 checkpoints)

**Files:**
- Create: `notebooks/06_verify.py`

**Interfaces:**
- Consumes: all of the above.
- Produces: pass/fail assertions for the four definition-of-done checkpoints. Alias-based, never version integers.

- [ ] **Step 1: Write `notebooks/06_verify.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Verify (minimum bar)
# MAGIC Runs the four definition-of-done checkpoints against the mandatory path.
# MAGIC Assertions are alias-based (re-runs bump version numbers).

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, pandas as pd, workshop_lib as wl
from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# 1) R2 in band
df = spark.table(DATA_TABLE).toPandas().sort_values("month")
r2 = wl.quick_fit_r2(df)
assert wl.R2_BAND[0] <= r2 <= wl.R2_BAND[1], f"R2 {r2:.3f} outside {wl.R2_BAND}"

# 2) forecaster registered
assert client.get_registered_model(FORECASTER_MODEL) is not None

# 3) @champion resolves (alias, not a literal version)
champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
assert champ is not None and champ.version is not None

# 4) chain returns a purchase decision
forecaster = mlflow.pyfunc.load_model(f"models:/{FORECASTER_MODEL}@champion")
optimizer = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}/latest")
latest = df.iloc[[-1]]
pp = float(forecaster.predict(latest[wl.DRIVERS + ["price"]])[0])
oi = latest[wl.ECON_COLS].copy(); oi.insert(0, "predicted_price", pp)
dec = optimizer.predict(oi)
assert dec.loc[0, "status"] == "optimal" and dec.loc[0, "purchase_qty"] == dec.loc[0, "purchase_qty"]
print("ALL 4 CHECKPOINTS PASSED")
```

- [ ] **Step 2: In-workspace verification**

Run All after `01 → 02 → 04 → 05`. Expected: `ALL 4 CHECKPOINTS PASSED`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/06_verify.py
git commit -m "Add 4-checkpoint verification harness notebook"
```

---

### Task 11: Optional PyTorch notebook

**Files:**
- Create: `notebooks/97_optional_pytorch.py`

**Interfaces:**
- Consumes: `DATA_TABLE`, `FORECASTER_MODEL`, `R2_THRESHOLD`, `SEED`, `workshop_lib.DRIVERS`.
- Produces: a new version of `FORECASTER_MODEL` registered + `@champion` re-pointed **iff** R² ≥ threshold. Not a dependency of any other notebook.

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

Xtr, ytr = to_t(train), torch.tensor(train["price_next_month"].to_numpy(), dtype=torch.float32).view(-1, 1)
Xte = to_t(test)

# COMMAND ----------
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x)

with mlflow.start_run(run_name="pytorch_mlp"):
    net = MLP(len(feats))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(400):
        opt.zero_grad(); loss = loss_fn(net(Xtr), ytr); loss.backward(); opt.step()
    preds = net(Xte).detach().numpy().ravel()
    r2 = float(r2_score(test["price_next_month"], preds))
    rmse = float(np.sqrt(mean_squared_error(test["price_next_month"], preds)))
    mlflow.log_params({"model_type": "torch_mlp", "epochs": 400, "seed": SEED})
    mlflow.log_metrics({"r2": r2, "rmse": rmse})
    info = mlflow.pytorch.log_model(net, "model")
    print(f"r2={r2:.3f} rmse={rmse:.2f}")

# COMMAND ----------
# Same gate as the sklearn lab.
from mlflow import MlflowClient
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(info.model_uri, FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate. Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f}). Not promoted.")
```

- [ ] **Step 2: In-workspace verification**

Run All independently. Expected: a `pytorch_mlp` run logged; gate prints PASS/FAIL; if PASS, a new `FORECASTER_MODEL` version exists. Confirm the end-to-end notebook still works regardless (loads whatever `@champion` points to).

- [ ] **Step 3: Commit**

```bash
git add notebooks/97_optional_pytorch.py
git commit -m "Add optional PyTorch notebook reusing the validation gate"
```

---

### Task 12: Optional Model Serving notebook + notebooks README

**Files:**
- Create: `notebooks/99_optional_serving.py`
- Create: `notebooks/README.md`

**Interfaces:**
- Consumes: `FORECASTER_MODEL`, `OPTIMIZER_MODEL`.
- Produces: a served endpoint for the forecaster (and notes for the optimizer endpoint); a README describing run order and the local test suite.

- [ ] **Step 1: Write `notebooks/99_optional_serving.py`**

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · Optional — Model Serving
# MAGIC Deploy registered models to a serving endpoint and query them. The Pyomo
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
# MAGIC When the endpoint is Ready, query it (drivers + price columns), or use the
# MAGIC Serving UI. To serve the Pyomo optimizer, create a second endpoint for
# MAGIC `OPTIMIZER_MODEL` — its `pip_requirements` already include `highspy`/`pyomo`.
```

- [ ] **Step 2: Write `notebooks/README.md`**

```markdown
# Workshop notebooks

Run order (mandatory path): `00_setup` → `01_generate_data` → `02_sklearn_baseline`
→ `04_pyomo_pyfunc` → `05_end_to_end` → `06_verify`.

Optional: `97_optional_pytorch`, `99_optional_serving`.

`workshop_lib.py` holds the pure, reusable logic (data generation, the Pyomo
optimizer, the PyFunc wrapper) and is unit-tested locally:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

Set the `catalog` and `schema` widgets in `_config` to your own Unity Catalog
target. All model references use the `@champion` alias, never version numbers.
```

- [ ] **Step 3: In-workspace verification**

Run `99_optional_serving.py` (after the forecaster is registered). Expected: endpoint creation starts; once Ready, a query returns a prediction. (Endpoint provisioning takes several minutes — acceptable for an optional lab.)

- [ ] **Step 4: Commit**

```bash
git add notebooks/99_optional_serving.py notebooks/README.md
git commit -m "Add optional serving notebook and notebooks README"
```

---

## Self-Review

**Spec coverage:**
- Customer-agnostic, synthetic data, fixed seed → Global Constraints + Task 2. ✓
- Two-act narrative (sklearn standard, Pyomo headline) → Tasks 7, 8. ✓
- Optional PyTorch, non-dependency → Task 11 (`97_`), Task 9 depends only on `@champion`. ✓
- Deterministic single-model gate (R² ≥ 0.6) → Tasks 7, 11. ✓
- UC widget-parameterized registry, alias-first → Tasks 5, 7, 9, 10. ✓
- Data: 96 months, time-ordered, 10 drivers, causal target, R²-band assert, `trend_up` derived-only, econ columns → Tasks 2, 6. ✓
- Pyomo HiGHS PyFunc, separate model, load & call → Tasks 3, 4, 8. ✓
- End-to-end chain + lineage beat → Task 9. ✓
- 4-checkpoint verification, alias-based → Task 10. ✓
- Optional serving with solver-in-container note → Task 12. ✓
- `.py` source format, `_config` via `%run` → Tasks 5–12. ✓

**Placeholder scan:** No "TBD"/"add error handling"-style gaps; every code step shows real content. The one open choice (optimizer alias vs. `/latest`) is called out explicitly in Task 8 with instruction to keep Task 10 consistent.

**Type consistency:** `generate_dataset`, `quick_fit_r2`, `solve_purchase`, `PurchaseOptimizerModel.predict` signatures match across Tasks 2–4 and their notebook consumers (Tasks 6–11). Constants `SEED`, `R2_THRESHOLD`, `R2_BAND`, `DRIVERS`, `ECON_COLS` are defined once in `workshop_lib` and re-exposed by `_config`. Model name constants `FORECASTER_MODEL`/`OPTIMIZER_MODEL`/`DATA_TABLE` defined in Task 5, consumed consistently thereafter.

**Out of scope (correctly excluded):** champion–challenger, classification fork, Automate & Monitor, MkDocs site (separate plan).
