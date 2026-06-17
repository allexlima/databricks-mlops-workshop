# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Shared config
# MAGIC `%run` this from every lab notebook. Edit CATALOG/SCHEMA once here and every
# MAGIC notebook inherits them. Reproducibility constants live in `workshop_lib`.

# COMMAND ----------
import workshop_lib as wl

# 👉 Set these to a Unity Catalog + schema you can write to. Edit once; every
# notebook picks it up via `%run ./_config`.
CATALOG = "main"
SCHEMA = "mlops_workshop"

EXPERIMENT_PATH = "/Shared/mlops_workshop"
FORECASTER_MODEL = f"{CATALOG}.{SCHEMA}.price_forecaster"
OPTIMIZER_MODEL = f"{CATALOG}.{SCHEMA}.purchase_optimizer"
DATA_TABLE = f"{CATALOG}.{SCHEMA}.commodity_monthly"
VOLUME = "workshop_files"                              # UC Volume (serverless-safe file storage)
CSV_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/commodity_monthly.csv"
SEED = wl.SEED
R2_THRESHOLD = wl.R2_THRESHOLD
print(f"catalog={CATALOG} schema={SCHEMA} forecaster={FORECASTER_MODEL}")
