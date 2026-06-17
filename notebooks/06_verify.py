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
