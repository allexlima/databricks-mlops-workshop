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
# MAGIC # 04 · End-to-end chain
# MAGIC Load the registered predictor (`@champion`) and optimizer (`@champion`),
# MAGIC then run drivers → predicted price → purchase decision on the latest month.
# MAGIC A once-manual monthly run now lives in a governed lineage.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 · Imports & registry setup
# MAGIC Point MLflow at Unity Catalog so model aliases resolve against the UC registry.

import mlflow
import pandas as pd
import workshop_lib as wl
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 · Champion guard
# MAGIC Fail immediately with a clear message if lab 02 hasn't promoted a champion yet,
# MAGIC rather than surfacing a raw registry error later in the chain.

try:
    client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
except Exception:
    raise RuntimeError(
        f"No @champion alias on {FORECASTER_MODEL}. Run 02_train_forecaster_sklearn "
        f"and confirm it passed the R2>={R2_THRESHOLD} gate before running this notebook."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 · Load both models by alias
# MAGIC Using `@champion` means this cell always picks up the latest promoted version —
# MAGIC no manual version numbers to update between runs.

df = spark.table(DATA_TABLE).toPandas().sort_values("month")
latest = df.iloc[[-1]]

forecaster = mlflow.pyfunc.load_model(f"models:/{FORECASTER_MODEL}@champion")
optimizer = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 · Run the chain: predict price → optimize → print
# MAGIC The forecaster estimates next-month commodity price; the optimizer uses that
# MAGIC estimate alongside economic context to recommend a purchase quantity.

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
