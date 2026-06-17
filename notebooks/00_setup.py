# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Installs deps, points MLflow at Unity Catalog, creates the catalog/schema
# MAGIC and experiment. Run once before the labs.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow
mlflow.set_registry_uri("databricks-uc")
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
mlflow.set_experiment(EXPERIMENT_PATH)
print("Setup complete.")
