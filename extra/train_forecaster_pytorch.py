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

df = spark.table(DATA_TABLE).toPandas().sort_values("month")
feats = wl.DRIVERS + ["price"]
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]

mu, sd = train[feats].mean(), train[feats].std(ddof=0)


def to_t(frame):
    return torch.tensor(((frame[feats] - mu) / sd).to_numpy(), dtype=torch.float32)


Xtr = to_t(train)
ytr = torch.tensor(train["price_next_month"].to_numpy(), dtype=torch.float32).view(-1, 1)
Xte = to_t(test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define MLP, train, and log to MLflow
# MAGIC A two-layer network (Linear → ReLU → Linear) trained with Adam for 400 epochs.
# MAGIC Everything — params, metrics, and the serialised model — is captured in a single
# MAGIC MLflow run so the artifact URI is available for the registration step below.

# COMMAND ----------


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


with mlflow.start_run(run_name="pytorch_mlp"):
    net = MLP(len(feats))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()

    preds = net(Xte).detach().numpy().ravel()
    r2 = float(r2_score(test["price_next_month"], preds))
    rmse = float(np.sqrt(mean_squared_error(test["price_next_month"], preds)))

    mlflow.log_params({"model_type": "torch_mlp", "epochs": 400, "seed": SEED})
    mlflow.log_metrics({"r2": r2, "rmse": rmse})
    info = mlflow.pytorch.log_model(net, name="model")
    print(f"r2={r2:.3f} rmse={rmse:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validation gate — same threshold as the sklearn lab
# MAGIC Reuses the shared `R2_THRESHOLD` constant: only promote to UC and alias as
# MAGIC `@champion` if the model meets the bar.

# COMMAND ----------

# Same gate as the sklearn lab.
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(info.model_uri, FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate. Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f}). Not promoted.")
