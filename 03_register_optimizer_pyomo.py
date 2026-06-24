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
# MAGIC # 03 · Pyomo optimizer as a custom PyFunc
# MAGIC A non-trainable operations-research model wrapped as MLflow PyFunc lives in
# MAGIC the exact same registry/lifecycle as any ML predictor — no special treatment required.

# COMMAND ----------

# MAGIC %md
# MAGIC **Setup** — load workshop constants and imports. `mlflow.set_experiment` ensures
# MAGIC all runs from this notebook land in the shared workshop experiment.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import mlflow
import pandas as pd
import workshop_lib as wl
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC **Build example input and run the optimizer** — creates a single-row DataFrame that
# MAGIC represents a purchase decision problem, then calls the solver to get a solution. TEST

# COMMAND ----------

# Paste the code for "Rodando o exemplo antes de registrar" here.
# Copy it from the workshop site → Lab 3 (Empacotar o otimizador como PyFunc), "Rodando o exemplo antes de registrar".

# COMMAND ----------

# MAGIC %md
# MAGIC **Feasibility check** — the solver must return "optimal" and the purchase quantity
# MAGIC must lie within demand/capacity bounds before we trust this model enough to register it.

# COMMAND ----------

# Paste the code for "O assert de viabilidade" here.
# Copy it from the workshop site → Lab 3 (Registrar e promover a @champion), "O assert de viabilidade".

# COMMAND ----------

# MAGIC %md
# MAGIC **Log, register, and promote** — packages the Pyomo model as an MLflow PyFunc,
# MAGIC ships `workshop_lib.py` as a code dependency so the solver is available at serving
# MAGIC time, then sets the `@champion` alias so the whole workshop loads models uniformly.

# COMMAND ----------

# Paste the code for "Fazer o log do modelo com code_paths" + "Definir o alias @champion"
# here (both blocks go in this one cell: o log_model, depois o set_registered_model_alias).
# Copy them from the workshop site → Lab 3 (Registrar e promover a @champion).

# COMMAND ----------

# MAGIC %md
# MAGIC **Reload by alias and predict** — proves the model is governed like any other:
# MAGIC load it by `@champion` URI and call `.predict()` on fresh input.

# COMMAND ----------

# Paste the code for "Carregar pelo alias e confirmar" here.
# Copy it from the workshop site → Lab 3 (Registrar e promover a @champion), "Carregar pelo alias e confirmar".
