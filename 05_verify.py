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
# MAGIC # 05 · Verify — Definition of Done
# MAGIC Four checkpoints that must all pass before a run is considered complete.
# MAGIC Assertions are alias-based so they survive version bumps without edits.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup — MLflow client and training data
# MAGIC Load the MlflowClient and pull the feature table into a sorted Pandas DataFrame so every checkpoint below works from the same snapshot.

# COMMAND ----------
import mlflow
import workshop_lib as wl
from mlflow import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

df = spark.table(DATA_TABLE).toPandas().sort_values("month")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checkpoint 1 — R² in band
# MAGIC Confirms the raw feature data still supports a reasonable linear fit; catches data-drift or schema changes before touching the registry.

# COMMAND ----------
r2 = wl.quick_fit_r2(df)
assert wl.R2_BAND[0] <= r2 <= wl.R2_BAND[1], f"R2 {r2:.3f} outside {wl.R2_BAND}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checkpoint 2 — Forecaster model is registered
# MAGIC Verifies the model name exists in Unity Catalog; fails fast if the training job never wrote to the registry.

# COMMAND ----------
assert client.get_registered_model(FORECASTER_MODEL) is not None

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checkpoint 3 — `@champion` alias resolves
# MAGIC Resolves the `@champion` alias (never a literal version integer) so promotion scripts and downstream consumers always point at the right version.

# COMMAND ----------
champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
assert champ is not None and champ.version is not None

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checkpoint 4 — End-to-end chain returns a purchase decision
# MAGIC Loads both champion models and runs a full inference pass on the most recent row; confirms the optimizer returns `status == "optimal"` with a non-null `purchase_qty`.

# COMMAND ----------
forecaster = mlflow.pyfunc.load_model(f"models:/{FORECASTER_MODEL}@champion")
optimizer = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
latest = df.iloc[[-1]]
pp = float(forecaster.predict(latest[wl.DRIVERS + ["price"]])[0])
oi = latest[wl.ECON_COLS].copy(); oi.insert(0, "predicted_price", pp)
dec = optimizer.predict(oi)
assert dec.loc[0, "status"] == "optimal" and pd.notna(dec.loc[0, "purchase_qty"])
print("ALL 4 CHECKPOINTS PASSED")
