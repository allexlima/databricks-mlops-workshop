# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02 · sklearn standard path
# MAGIC Train → track → evaluate against a fixed bar → register and promote to
# MAGIC `@champion` only if it passes. The everyday MLOps loop.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, numpy as np, workshop_lib as wl
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

df = spark.table(DATA_TABLE).toPandas().sort_values("month")
feats = wl.DRIVERS + ["price"]
cut = int(len(df) * 0.8)                     # time-respecting split, no shuffle
train, test = df.iloc[:cut], df.iloc[cut:]

# COMMAND ----------
with mlflow.start_run(run_name="sklearn_gbr") as run:
    model = GradientBoostingRegressor(random_state=SEED)
    model.fit(train[feats], train["price_next_month"])
    preds = model.predict(test[feats])
    rmse = float(np.sqrt(mean_squared_error(test["price_next_month"], preds)))
    mae = float(mean_absolute_error(test["price_next_month"], preds))
    r2 = float(r2_score(test["price_next_month"], preds))
    mlflow.log_params({"model_type": "GradientBoostingRegressor", "seed": SEED})
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.sklearn.log_model(
        model, name="model",
        serialization_format="cloudpickle",   # portable across envs (avoids skops)
        input_example=test[feats].head(3),
        signature=mlflow.models.infer_signature(test[feats], preds),
    )
    print(f"r2={r2:.3f} rmse={rmse:.2f}")

# COMMAND ----------
# Validation gate: register + promote only if the model clears the bar.
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate (r2={r2:.3f} >= {R2_THRESHOLD}). Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f} < {R2_THRESHOLD}). Not promoted.")
