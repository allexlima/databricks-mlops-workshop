# Databricks notebook source
# MAGIC %md
# MAGIC # Shared config
# MAGIC `%run` this from every lab notebook. Widgets + names + reproducibility
# MAGIC constants (single source of truth lives in `workshop_lib`).

# COMMAND ----------
import workshop_lib as wl

dbutils.widgets.text("catalog", "main", "Unity Catalog")
dbutils.widgets.text("schema", "mlops_workshop", "Schema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

EXPERIMENT_PATH = "/Shared/mlops_workshop"
FORECASTER_MODEL = f"{CATALOG}.{SCHEMA}.price_forecaster"
OPTIMIZER_MODEL = f"{CATALOG}.{SCHEMA}.purchase_optimizer"
DATA_TABLE = f"{CATALOG}.{SCHEMA}.commodity_monthly"
SEED = wl.SEED
R2_THRESHOLD = wl.R2_THRESHOLD
print(f"catalog={CATALOG} schema={SCHEMA} forecaster={FORECASTER_MODEL}")
