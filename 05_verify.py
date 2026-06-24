# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "pyomo",
#   "highspy",
# ]
# ///

# COMMAND ----------
# MAGIC %md
# MAGIC # 05 · Verify — Definition of Done
# MAGIC Four checkpoints that must all pass before a run is considered complete.
# MAGIC Assertions are alias-based so they survive version bumps without edits.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup — MLflow client and training data
# MAGIC Load the MlflowClient and pull the feature table into a sorted Pandas DataFrame so every checkpoint below works from the same snapshot.

# COMMAND ----------
import mlflow
import workshop_lib as wl
from mlflow import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

df = spark.table(DATA_TABLE).toPandas().sort_values("month")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checkpoint 1 — R² in band
# MAGIC Confirms the raw feature data still supports a reasonable linear fit; catches data-drift or schema changes before touching the registry.

# COMMAND ----------
# Paste the code for "Checkpoint 1: R² dentro da faixa esperada" here.
# Copy it from the workshop site → Lab 5 (Verificação), "Checkpoint 1".

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checkpoint 2 — Forecaster model is registered
# MAGIC Verifies the model name exists in Unity Catalog; fails fast if the training job never wrote to the registry.

# COMMAND ----------
# Paste the code for "Checkpoint 2: price_forecaster está registrado no Unity Catalog" here.
# Copy it from the workshop site → Lab 5 (Verificação), "Checkpoint 2".

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checkpoint 3 — `@champion` alias resolves
# MAGIC Resolves the `@champion` alias (never a literal version integer) so promotion scripts and downstream consumers always point at the right version.

# COMMAND ----------
# Paste the code for "Checkpoint 3: O alias @champion resolve" here.
# Copy it from the workshop site → Lab 5 (Verificação), "Checkpoint 3".

# COMMAND ----------
# MAGIC %md
# MAGIC ## Checkpoint 4 — End-to-end chain returns a purchase decision
# MAGIC Loads both champion models and runs a full inference pass on the most recent row; confirms the optimizer returns `status == "optimal"` with a non-null `purchase_qty`.

# COMMAND ----------
# Paste the code for "Checkpoint 4: A cadeia ponta a ponta retorna uma decisão de compra" here.
# Copy it from the workshop site → Lab 5 (Verificação), "Checkpoint 4".
