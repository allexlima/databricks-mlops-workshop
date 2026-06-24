# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "torch",
# ]
# ///

# COMMAND ----------

# MAGIC %md
# MAGIC # Optional — PyTorch Forecaster (same lifecycle, different flavor)
# MAGIC This notebook is **optional and is NOT a dependency of the end-to-end workshop chain.**
# MAGIC It runs the same data, time split, and validation gate as the sklearn lab — but replaces
# MAGIC the gradient-boosted model with a small fully-connected MLP, so you can compare how the
# MAGIC MLflow-based lifecycle looks with a deep-learning framework.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup — config, imports, and seeds
# MAGIC Load the shared workshop config (`_config` lives one level up), then import the
# MAGIC ML stack and fix the random seed for reproducible training.

# COMMAND ----------

# MAGIC %run ../_config

# COMMAND ----------

import mlflow
import numpy as np
import torch
import torch.nn as nn
import workshop_lib as wl
from mlflow import MlflowClient
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
torch.manual_seed(SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load data, apply the time split, and build tensors
# MAGIC The 80/20 chronological split mirrors the sklearn lab exactly — features are
# MAGIC z-scored using **only training statistics** to avoid data leakage.

# COMMAND ----------

# Paste the code for "Dados, split e normalização: idênticos ao forecaster sklearn" here.
# Copy it from the workshop site → Forecaster em PyTorch, "Dados, split e normalização".

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define MLP, train, and log to MLflow
# MAGIC A two-layer network (Linear → ReLU → Linear) trained with Adam for 400 epochs.
# MAGIC Everything — params, metrics, and the serialised model — is captured in a single
# MAGIC MLflow run so the artifact URI is available for the registration step below.

# COMMAND ----------


# Paste the code for "O modelo: MLP de duas camadas" + "Tracking e log: dentro de um
# único mlflow.start_run" here (both blocks go in this one cell: the MLP class, then the
# training run). Copy them from the workshop site → Forecaster em PyTorch.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validation gate — same threshold as the sklearn lab
# MAGIC Reuses the shared `R2_THRESHOLD` constant: only promote to UC and alias as
# MAGIC `@champion` if the model meets the bar.

# COMMAND ----------

# Paste the code for "Validation gate e promoção para @champion" here.
# Copy it from the workshop site → Forecaster em PyTorch, "Validation gate".
