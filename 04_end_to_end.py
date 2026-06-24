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
# MAGIC # 04 · End-to-end chain
# MAGIC Load the registered predictor (`@champion`) and optimizer (`@champion`),
# MAGIC then run drivers → predicted price → purchase decision on the latest month.
# MAGIC A once-manual monthly run now lives in a governed lineage.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 · Imports & registry setup
# MAGIC Point MLflow at Unity Catalog so model aliases resolve against the UC registry.

import mlflow
import pandas as pd
import workshop_lib as wl
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 · Champion guard
# MAGIC Fail immediately with a clear message if lab 02 hasn't promoted a champion yet,
# MAGIC rather than surfacing a raw registry error later in the chain.

# Paste the code for "O champion guard" here.
# Copy it from the workshop site → Lab 4 (Carregar os modelos por @champion), "O champion guard".

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 · Load both models by alias
# MAGIC Using `@champion` means this cell always picks up the latest promoted version —
# MAGIC no manual version numbers to update between runs.

# Paste the code for "Carregando os dois modelos por alias" here.
# Copy it from the workshop site → Lab 4 (Carregar os modelos por @champion), "Carregando os dois modelos por alias".

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 · Run the chain: predict price → optimize → print
# MAGIC The forecaster estimates next-month commodity price; the optimizer uses that
# MAGIC estimate alongside economic context to recommend a purchase quantity.

# Paste the code for "A cadeia em quatro passos" (bloco completo) here.
# Copy it from the workshop site → Lab 4 (Rodar a cadeia e obter a decisão), "A cadeia em quatro passos".

# COMMAND ----------
# MAGIC %md
# MAGIC ### Governance & lineage
# MAGIC Open **Catalog Explorer → the Delta table → Lineage** to see table →
# MAGIC experiment → registered models. The ungoverned monthly manual run now has
# MAGIC reproducible inputs, tracked runs, registered models, and lineage.
