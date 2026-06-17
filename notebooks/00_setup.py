# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Points MLflow at Unity Catalog and creates the catalog/schema/volume and
# MAGIC experiment. Run once before the labs.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Dependencies
# MAGIC **Serverless (recommended):** dependencies are declared in each notebook's
# MAGIC serverless **Environment** and persisted as PEP 723 metadata in the source —
# MAGIC nothing to install, just run the cells.
# MAGIC
# MAGIC **Classic / ML cluster:** the PEP 723 environment is ignored. Run this once at
# MAGIC the top of each notebook (or install the libs on the cluster):
# MAGIC ```
# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python
# MAGIC ```

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow
mlflow.set_registry_uri("databricks-uc")

# The catalog must already exist — creating catalogs is privileged and depends on
# managed-location / Default-Storage settings. Set CATALOG (in _config) to one you
# can create schemas in. We only create the schema inside it.
catalogs = [r[0] for r in spark.sql("SHOW CATALOGS").collect()]
assert CATALOG in catalogs, (
    f"Catalog '{CATALOG}' not found. Set the 'catalog' widget to an existing Unity "
    f"Catalog you can write to (available: {catalogs})."
)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")  # serverless-safe file storage
mlflow.set_experiment(EXPERIMENT_PATH)
print(f"Setup complete. Using {CATALOG}.{SCHEMA}.")
