# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Pyomo optimizer as a custom PyFunc (the headline)
# MAGIC A non-trainable operations-research model, wrapped as MLflow PyFunc, lives
# MAGIC in the exact same registry/lifecycle as the ML predictor.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import mlflow, pandas as pd, workshop_lib as wl
from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

example = pd.DataFrame([{
    "predicted_price": 210.0, "holding_cost": 1.0, "purchase_cost": 200.0,
    "demand": 100.0, "capacity": 200.0, "budget": 30000.0,
}])
sample_out = wl.PurchaseOptimizerModel().predict(None, example)

# Inline sanity assert: the solve must be feasible and meet demand within capacity.
row = sample_out.iloc[0]
assert row["status"] == "optimal", f"optimizer infeasible on example: {row.to_dict()}"
assert 100.0 - 1e-6 <= row["purchase_qty"] <= 200.0 + 1e-6, f"qty out of bounds: {row['purchase_qty']}"
print(sample_out)

# COMMAND ----------
signature = mlflow.models.infer_signature(example, sample_out)
client = MlflowClient()
with mlflow.start_run(run_name="pyomo_optimizer"):
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=wl.PurchaseOptimizerModel(),
        code_paths=["./workshop_lib.py"],          # ship the optimizer logic
        pip_requirements=["pyomo>=6.7", "highspy>=1.7", "pandas>=2.0"],
        input_example=example,
        signature=signature,
        registered_model_name=OPTIMIZER_MODEL,
    )
# Promote by alias so the whole workshop loads uniformly by @champion.
client.set_registered_model_alias(OPTIMIZER_MODEL, "champion", info.registered_model_version)

# COMMAND ----------
# Load BY ALIAS and call — proves it is governed like any model.
opt = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
print(opt.predict(example))
