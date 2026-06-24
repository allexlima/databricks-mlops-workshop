# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///

# MAGIC %md
# MAGIC # 01 · Generate the dataset
# MAGIC One shared, reproducible, time-ordered synthetic commodity dataset used by every
# MAGIC notebook in this workshop. All models read from this single source so comparisons
# MAGIC are apples-to-apples. We also assert the signal is learnable-but-not-trivial.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1 · Load config and generate the raw dataset
# MAGIC `_config` defines workspace-level constants (`SEED`, `DATA_TABLE`, `CSV_PATH`).
# MAGIC `wl.generate_dataset` returns a pandas DataFrame with 96 monthly rows, time-ordered.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

import workshop_lib as wl

# Paste the code for "Passo 1 · Carregar a configuração e gerar o DataFrame bruto" here.
# Copy it from the workshop site → Lab 1, "Passo 1".

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 · Data dictionary and sanity checks
# MAGIC Descriptive stats confirm the scale of each feature; correlations show which
# MAGIC drivers carry the most signal toward `price_next_month`.

# COMMAND ----------

# Paste the code for "Passo 2 · Dicionário de dados e verificações de sanidade" here.
# Copy it from the workshop site → Lab 1, "Passo 2".

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 · Naive baseline vs. quick-model R²
# MAGIC A lag-1 naive forecast (predict today's price for tomorrow) sets the floor.
# MAGIC A quick regularised model sets the ceiling. Everything built in later notebooks
# MAGIC should land between these two values.

# COMMAND ----------

# Paste the code for "Passo 3 · Baseline ingênuo vs. R² do modelo rápido" here.
# Copy it from the workshop site → Lab 1, "Passo 3".

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4 · Signal-quality gate
# MAGIC This assert fails loudly if the synthetic signal is mis-tuned — too weak means
# MAGIC no model can beat the naive baseline; too strong means every model looks trivial.

# COMMAND ----------

# Paste the code for "Passo 4 · Portão de qualidade do sinal (inline assert)" here.
# Copy it from the workshop site → Lab 1, "Passo 4".

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5 · Persist: Delta table + CSV
# MAGIC Write to a Unity Catalog managed Delta table (governed, query-able) and a CSV on
# MAGIC a UC Volume (portable, serverless-safe) so every notebook can read either format.

# COMMAND ----------

# Paste the code for "Passo 5 · Persistir como tabela Delta + CSV em um UC Volume" here.
# Copy it from the workshop site → Lab 1, "Passo 5".
