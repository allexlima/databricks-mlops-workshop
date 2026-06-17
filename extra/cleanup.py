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
# MAGIC This notebook **deletes everything by default.** To stop it, set `CONFIRM = False`
# MAGIC (the assert below then aborts before any deletion). Set `DROP_SCHEMA = False` to keep
# MAGIC the schema and only remove the models, table, and volume.

# COMMAND ----------

CONFIRM = True       # 👉 set to False to abort before any deletion
DROP_SCHEMA = True   # 👉 set to False to keep the schema (drop only models/table/volume)

assert CONFIRM, "CONFIRM is False — aborting teardown. Set CONFIRM = True to delete."

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
# MAGIC Drop the entire schema with `CASCADE`, removing anything else left behind.
# MAGIC On by default (`DROP_SCHEMA = True`); set it to `False` above if the catalog or schema
# MAGIC is shared with other users or workshops.

# COMMAND ----------

if DROP_SCHEMA:
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
    print(f"Dropped schema {CATALOG}.{SCHEMA}.")
print("Cleanup complete.")
