# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "torch",
# ]
# ///
# MAGIC %md
# MAGIC # 97 · Optional — PyTorch (same lifecycle, different flavor)
# MAGIC The SAME data/split through a small MLP, reusing the SAME validation gate.
# MAGIC Optional: the end-to-end chain does not depend on this notebook.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, numpy as np, torch, torch.nn as nn, workshop_lib as wl
from mlflow import MlflowClient
from sklearn.metrics import r2_score, mean_squared_error
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
torch.manual_seed(SEED)

df = spark.table(DATA_TABLE).toPandas().sort_values("month")
feats = wl.DRIVERS + ["price"]
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]
mu, sd = train[feats].mean(), train[feats].std(ddof=0)

def to_t(frame):
    return torch.tensor(((frame[feats] - mu) / sd).to_numpy(), dtype=torch.float32)

Xtr = to_t(train); ytr = torch.tensor(train["price_next_month"].to_numpy(), dtype=torch.float32).view(-1, 1)
Xte = to_t(test)

# COMMAND ----------
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x)

with mlflow.start_run(run_name="pytorch_mlp"):
    net = MLP(len(feats)); opt = torch.optim.Adam(net.parameters(), lr=0.01); loss_fn = nn.MSELoss()
    for _ in range(400):
        opt.zero_grad(); loss = loss_fn(net(Xtr), ytr); loss.backward(); opt.step()
    preds = net(Xte).detach().numpy().ravel()
    r2 = float(r2_score(test["price_next_month"], preds))
    rmse = float(np.sqrt(mean_squared_error(test["price_next_month"], preds)))
    mlflow.log_params({"model_type": "torch_mlp", "epochs": 400, "seed": SEED})
    mlflow.log_metrics({"r2": r2, "rmse": rmse})
    info = mlflow.pytorch.log_model(net, name="model")
    print(f"r2={r2:.3f} rmse={rmse:.2f}")

# COMMAND ----------
# Same gate as the sklearn lab.
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(info.model_uri, FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate. Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f}). Not promoted.")
