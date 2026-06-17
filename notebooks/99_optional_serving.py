# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · Optional — Model Serving
# MAGIC Deploy a registered model to a serving endpoint and query it. The Pyomo
# MAGIC endpoint works because `highspy` is pip-installable inside the container.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
from mlflow.deployments import get_deploy_client
from mlflow import MlflowClient
client = MlflowClient()
deploy = get_deploy_client("databricks")

champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
endpoint = "mlops-workshop-forecaster"
deploy.create_endpoint(
    name=endpoint,
    config={"served_entities": [{
        "entity_name": FORECASTER_MODEL, "entity_version": champ.version,
        "workload_size": "Small", "scale_to_zero_enabled": True,
    }]},
)
print(f"Creating endpoint {endpoint} for {FORECASTER_MODEL} v{champ.version}.")

# COMMAND ----------
# MAGIC %md
# MAGIC When Ready, query it (drivers + price columns) via the Serving UI or REST.
# MAGIC To serve the Pyomo optimizer, create a second endpoint for `OPTIMIZER_MODEL`
# MAGIC — its `pip_requirements` already include `highspy`/`pyomo`.
