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
