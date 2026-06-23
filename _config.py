# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Shared config
# MAGIC `%run` this from every lab notebook. Edit only **CATALOG** once here — SCHEMA,
# MAGIC experiment path, and model names are derived automatically from the running
# MAGIC user, giving each participant a fully isolated workspace. Reproducibility
# MAGIC constants live in `workshop_lib`.

# COMMAND ----------

# DBTITLE 1,Cell 2
import os
import sys

# Make the repo root importable whether this config is %run from the root
# (mandatory notebooks) or a subfolder like extra/ (optional notebooks).
for _p in (os.getcwd(), os.path.dirname(os.getcwd())):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import workshop_lib as wl

# 👉 Set CATALOG to a Unity Catalog you can create schemas in. Edit once here.
# SCHEMA, EXPERIMENT_PATH, SERVING_ENDPOINT, and all model/data paths are
# derived automatically from the current user so every participant gets a
# fully isolated workspace — no per-user config edits needed.
CATALOG = "main"

# Derive a safe identifier from the running user's email.
_current_user = spark.sql("SELECT current_user()").collect()[0][0]
_user_suffix  = _current_user.split("@")[0].replace(".", "_").replace("-", "_")

SCHEMA           = f"mlops_workshop_{_user_suffix}"
EXPERIMENT_PATH  = f"/Users/{_current_user}/mlops_workshop"
SERVING_ENDPOINT = f"mlops-workshop-forecaster-{_user_suffix}"

FORECASTER_MODEL = f"{CATALOG}.{SCHEMA}.price_forecaster"
OPTIMIZER_MODEL  = f"{CATALOG}.{SCHEMA}.purchase_optimizer"
DATA_TABLE       = f"{CATALOG}.{SCHEMA}.commodity_monthly"
VOLUME           = "workshop_files"                    # UC Volume (serverless-safe file storage)
CSV_PATH         = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/commodity_monthly.csv"
SEED             = wl.SEED
R2_THRESHOLD     = wl.R2_THRESHOLD

print(f"user={_current_user!r}")
print(f"catalog={CATALOG}  schema={SCHEMA}")
print(f"experiment={EXPERIMENT_PATH}")
print(f"forecaster={FORECASTER_MODEL}")
