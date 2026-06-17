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

    # Next-month price: current drivers (via base) + lagged price + modest noise.
    # No shuffle/shift — the current row's drivers forecast next month's price.
    noise = rng.normal(0, 12, n)                 # tuned so quick_fit_r2 centers ~0.75 in band
    df["price_next_month"] = base + 0.3 * (df["price"].to_numpy() - base) + noise
    df["trend_up"] = df["price_next_month"] > df["price"]

    df["holding_cost"] = rng.uniform(0.5, 2.0, n)
    df["purchase_cost"] = df["price"] * rng.uniform(0.9, 1.1, n)
    df["demand"] = rng.uniform(80, 140, n)
    df["capacity"] = rng.uniform(150, 220, n)
    df["budget"] = rng.uniform(20000, 40000, n)
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

    opt = Highs()
    opt.config.load_solution = False             # don't raise when infeasible
    result = opt.solve(m)
    status = str(result.termination_condition).lower()
    if "optimal" not in status:
        return {"purchase_qty": float("nan"), "total_cost": float("nan"), "status": "infeasible"}
    result.solution_loader.load_vars()
    return {"purchase_qty": float(pyo.value(m.q)),
            "total_cost": float(pyo.value(m.obj)), "status": "optimal"}


class PurchaseOptimizerModel(mlflow.pyfunc.PythonModel):
    """Wraps the Pyomo optimizer as an MLflow PyFunc so a non-trainable OR model
    lives in the same registry/lifecycle as the ML predictor."""

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        rows = [
            solve_purchase(
                predicted_price=r["predicted_price"], holding_cost=r["holding_cost"],
                purchase_cost=r["purchase_cost"], demand=r["demand"],
                capacity=r["capacity"], budget=r["budget"],
            )
            for _, r in model_input.iterrows()
        ]
        return pd.DataFrame(rows, columns=["purchase_qty", "total_cost", "status"])
