# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///

# COMMAND ----------

# MAGIC %md
# MAGIC # Optional — Cleanup (Teardown)
# MAGIC **DESTRUCTIVE.** This notebook permanently removes every resource the workshop created:
# MAGIC serving endpoint, registered models, MLflow experiment, Delta table, volume, and
# MAGIC (optionally) the entire schema. Run only when you are finished with the workshop.

# COMMAND ----------

# MAGIC %md
# MAGIC Load shared config variables (`CATALOG`, `SCHEMA`, `DATA_TABLE`, `FORECASTER_MODEL`,
# MAGIC `OPTIMIZER_MODEL`, `EXPERIMENT_PATH`, `VOLUME`) used by all cells below.

# COMMAND ----------

# MAGIC %run ../_config

# COMMAND ----------

# MAGIC %md
# MAGIC Set the two widgets, then confirm teardown is intentional.
# MAGIC **You must change `confirm` to `yes` — this guard prevents accidental teardown.**

# COMMAND ----------

dbutils.widgets.dropdown("confirm", "no", ["no", "yes"], "Confirm teardown")
dbutils.widgets.dropdown("drop_schema", "no", ["no", "yes"], "Also drop schema (CASCADE)")
assert dbutils.widgets.get("confirm") == "yes", \
    "Set the 'confirm' widget to 'yes' to delete workshop resources."

# COMMAND ----------

# MAGIC %md
# MAGIC Delete the Model Serving endpoint created during the optional serving lab.
# MAGIC The `try/except` is intentional — the endpoint may not exist if that lab was skipped.

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
from mlflow.deployments import get_deploy_client

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    get_deploy_client("databricks").delete_endpoint("mlops-workshop-forecaster")
    print("Deleted serving endpoint.")
except Exception as e:
    print(f"No serving endpoint to delete ({e}).")

# COMMAND ----------

# MAGIC %md
# MAGIC Delete both Unity Catalog registered models (`FORECASTER_MODEL` and `OPTIMIZER_MODEL`),
# MAGIC including all versions. Each deletion is wrapped independently so one missing model
# MAGIC does not block the other.

# COMMAND ----------

for name in (FORECASTER_MODEL, OPTIMIZER_MODEL):
    try:
        client.delete_registered_model(name)
        print(f"Deleted registered model {name}.")
    except Exception as e:
        print(f"No model {name} ({e}).")

# COMMAND ----------

# MAGIC %md
# MAGIC Delete the MLflow experiment and all its runs.
# MAGIC The lookup-by-name step gracefully handles the case where the experiment was never created.

# COMMAND ----------

try:
    exp = client.get_experiment_by_name(EXPERIMENT_PATH)
    if exp:
        mlflow.delete_experiment(exp.experiment_id)
        print(f"Deleted experiment {EXPERIMENT_PATH}.")
except Exception as e:
    print(f"No experiment to delete ({e}).")

# COMMAND ----------

# MAGIC %md
# MAGIC Drop the Delta table and the CSV volume.
# MAGIC `IF EXISTS` makes both statements safe to re-run even if they were already removed.

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {DATA_TABLE}")
spark.sql(f"DROP VOLUME IF EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Dropped {DATA_TABLE} and volume {VOLUME}.")

# COMMAND ----------

# MAGIC %md
# MAGIC **Optional:** drop the entire schema with `CASCADE`, removing anything else left behind.
# MAGIC Off by default (`drop_schema = no`) because the catalog or schema may be shared with
# MAGIC other users or workshops.

# COMMAND ----------

if dbutils.widgets.get("drop_schema") == "yes":
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
    print(f"Dropped schema {CATALOG}.{SCHEMA}.")
print("Cleanup complete.")
