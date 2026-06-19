# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "pyomo",
#   "highspy",
# ]
# ///


# COMMAND ----------

# MAGIC %md
# MAGIC # 03 · Pyomo optimizer as a custom PyFunc
# MAGIC A non-trainable operations-research model wrapped as MLflow PyFunc lives in
# MAGIC the exact same registry/lifecycle as any ML predictor — no special treatment required.

# COMMAND ----------

# MAGIC %md
# MAGIC **Setup** — load workshop constants and imports. `mlflow.set_experiment` ensures
# MAGIC all runs from this notebook land in the shared workshop experiment.

# COMMAND ----------

# DBTITLE 1,Install Pyomo and HiGHS
# MAGIC %pip install pyomo>=6.7 highspy>=1.7 -q
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import mlflow
import pandas as pd
import workshop_lib as wl
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC **Build example input and run the optimizer** — creates a single-row DataFrame that
# MAGIC represents a purchase decision problem, then calls the solver to get a solution. TEST

# COMMAND ----------

example = pd.DataFrame([{
    "predicted_price": 210.0,
    "holding_cost": 1.0,
    "purchase_cost": 200.0,
    "demand": 100.0,
    "capacity": 200.0,
    "budget": 30000.0,
}])

sample_out = wl.PurchaseOptimizerModel().predict(None, example)
print(sample_out)

# COMMAND ----------

# MAGIC %md
# MAGIC **Feasibility check** — the solver must return "optimal" and the purchase quantity
# MAGIC must lie within demand/capacity bounds before we trust this model enough to register it.

# COMMAND ----------

row = sample_out.iloc[0]
assert row["status"] == "optimal", f"optimizer infeasible on example: {row.to_dict()}"
assert 100.0 - 1e-6 <= row["purchase_qty"] <= 200.0 + 1e-6, f"qty out of bounds: {row['purchase_qty']}"

# COMMAND ----------

# MAGIC %md
# MAGIC **Log, register, and promote** — packages the Pyomo model as an MLflow PyFunc,
# MAGIC ships `workshop_lib.py` as a code dependency so the solver is available at serving
# MAGIC time, then sets the `@champion` alias so the whole workshop loads models uniformly.

# COMMAND ----------

signature = mlflow.models.infer_signature(example, sample_out)
client = MlflowClient()

with mlflow.start_run(run_name="pyomo_optimizer"):
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=wl.PurchaseOptimizerModel(),
        code_paths=["./workshop_lib.py"],
        pip_requirements=["pyomo>=6.7", "highspy>=1.7", "pandas>=2.0"],
        input_example=example,
        signature=signature,
        registered_model_name=OPTIMIZER_MODEL,
    )

client.set_registered_model_alias(OPTIMIZER_MODEL, "champion", info.registered_model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC **Reload by alias and predict** — proves the model is governed like any other:
# MAGIC load it by `@champion` URI and call `.predict()` on fresh input.

# COMMAND ----------

opt = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
print(opt.predict(example))
