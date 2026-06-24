# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Optional — Model Serving
# MAGIC
# MAGIC Deploy the champion forecaster to a real-time serving endpoint and query it.
# MAGIC This notebook is **optional** — provisioning a new endpoint typically takes a
# MAGIC few minutes.  The Pyomo optimizer can be served the same way because `highspy`
# MAGIC is pip-installable inside the container.

# COMMAND ----------

# MAGIC %run ../_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the champion model version
# MAGIC
# MAGIC Look up whichever version is currently aliased `champion` in Unity Catalog so
# MAGIC the endpoint always pins to the exact registered version, not just a name.

# COMMAND ----------

# Paste the code for "Resolvendo a versão exata pelo alias" here.
# Copy it from the workshop site → Model Serving, "Resolvendo a versão exata pelo alias".

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the serving endpoint
# MAGIC
# MAGIC Provision a CPU-based endpoint with scale-to-zero enabled to keep costs low
# MAGIC when the endpoint is idle between demo runs.

# COMMAND ----------

# Paste the code for "Criando o endpoint com scale-to-zero" here.
# Copy it from the workshop site → Model Serving, "Criando o endpoint com scale-to-zero".

# COMMAND ----------

# MAGIC %md
# MAGIC When Ready, query it (drivers + price columns) via the Serving UI or REST.
# MAGIC To serve the Pyomo optimizer, create a second endpoint for `OPTIMIZER_MODEL`
# MAGIC — its `pip_requirements` already include `highspy`/`pyomo`.
