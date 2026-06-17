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

df = wl.generate_dataset(seed=SEED)  # 96 monthly rows, time-ordered
display(df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 · Data dictionary and sanity checks
# MAGIC Descriptive stats confirm the scale of each feature; correlations show which
# MAGIC drivers carry the most signal toward `price_next_month`.

# COMMAND ----------

print(df.describe())
print("\nDriver correlation with target:")
print(df[wl.DRIVERS].corrwith(df["price_next_month"]).sort_values())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 · Naive baseline vs. quick-model R²
# MAGIC A lag-1 naive forecast (predict today's price for tomorrow) sets the floor.
# MAGIC A quick regularised model sets the ceiling. Everything built in later notebooks
# MAGIC should land between these two values.

# COMMAND ----------

from sklearn.metrics import r2_score

cut = int(len(df) * 0.8)
naive_r2 = r2_score(df["price_next_month"].iloc[cut:], df["price"].iloc[cut:])
model_r2 = wl.quick_fit_r2(df)
print(f"naive R2={naive_r2:.3f}  quick-model R2={model_r2:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4 · Signal-quality gate
# MAGIC This assert fails loudly if the synthetic signal is mis-tuned — too weak means
# MAGIC no model can beat the naive baseline; too strong means every model looks trivial.

# COMMAND ----------

lo, hi = wl.R2_BAND
assert lo <= model_r2 <= hi, f"R2 {model_r2:.3f} outside band {wl.R2_BAND}"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5 · Persist: Delta table + CSV
# MAGIC Write to a Unity Catalog managed Delta table (governed, query-able) and a CSV on
# MAGIC a UC Volume (portable, serverless-safe) so every notebook can read either format.

# COMMAND ----------

spark.createDataFrame(df).write.mode("overwrite").saveAsTable(DATA_TABLE)
df.to_csv(CSV_PATH, index=False)
print(f"Wrote {DATA_TABLE} and {CSV_PATH}.")
