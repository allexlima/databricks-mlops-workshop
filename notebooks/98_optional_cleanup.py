# Databricks notebook source
# MAGIC %md
# MAGIC # 98 · Optional — Cleanup (teardown)
# MAGIC Removes everything the workshop created: serving endpoint, registered
# MAGIC models, the experiment, the Delta table (and optionally the schema).
# MAGIC **Destructive.** Set the `confirm` widget to `yes` to proceed.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

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

# 4) Data table + CSV volume.
spark.sql(f"DROP TABLE IF EXISTS {DATA_TABLE}")
spark.sql(f"DROP VOLUME IF EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Dropped {DATA_TABLE} and volume {VOLUME}.")

# COMMAND ----------
# Optional: drop the whole schema (removes anything left behind). Off by default
# because the catalog/schema may be shared.
if dbutils.widgets.get("drop_schema") == "yes":
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
    print(f"Dropped schema {CATALOG}.{SCHEMA}.")
print("Cleanup complete.")
