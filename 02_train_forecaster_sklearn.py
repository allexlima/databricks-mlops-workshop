# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///


# COMMAND ----------

# MAGIC %md
# MAGIC # 02 · sklearn standard path
# MAGIC Train → track → evaluate against a fixed bar → register and promote to
# MAGIC `@champion` only if it passes. The everyday MLOps loop.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports & experiment setup
# MAGIC Load libraries and point MLflow at the Unity Catalog registry and the per-user experiment so every run lands in the right place.

# COMMAND ----------

import mlflow
import numpy as np
import workshop_lib as wl
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data & time-respecting split
# MAGIC Sort by month and cut at 80 % — no shuffle — so the test set is always the most-recent observations and the model cannot peek at the future.

# COMMAND ----------

# Paste the code for "Passo 2 · Divisão treino/teste que respeita o tempo" here.
# Copy it from the workshop site → Lab 2 (Tracking com MLflow), "Passo 2".

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train & log MLflow run
# MAGIC Fit a `GradientBoostingRegressor`, compute held-out metrics, and log params, metrics, and the serialised model in a single atomic run.

# COMMAND ----------

# Paste the code for "Passo 3 · Treinar e logar um run completo no MLflow" here.
# Copy it from the workshop site → Lab 2 (Tracking com MLflow), "Passo 3".

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation gate — promote to @champion
# MAGIC Only models that clear R² ≥ threshold get registered and aliased to `@champion`; anything below is logged but left unregistered.

# COMMAND ----------

# Paste the code for "Passo 4 · Validation gate" here.
# Copy it from the workshop site → Lab 2 (Validation gate e registro), "Passo 4".
