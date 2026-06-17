# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Generate the dataset
# MAGIC One shared, reproducible, time-ordered synthetic commodity dataset. All
# MAGIC models read from this. We assert the signal is learnable-but-not-trivial.

# COMMAND ----------
# MAGIC %run ./_config

# COMMAND ----------
import workshop_lib as wl
df = wl.generate_dataset(seed=SEED)        # 96 monthly rows, time-ordered
display(df.head(10))

# COMMAND ----------
# MAGIC %md ## Data dictionary + sanity checks
# COMMAND ----------
print(df.describe())
print("\nDriver correlation with target:")
print(df[wl.DRIVERS].corrwith(df["price_next_month"]).sort_values())

from sklearn.metrics import r2_score
cut = int(len(df) * 0.8)
naive_r2 = r2_score(df["price_next_month"].iloc[cut:], df["price"].iloc[cut:])
model_r2 = wl.quick_fit_r2(df)
print(f"naive R2={naive_r2:.3f}  quick-model R2={model_r2:.3f}")

# COMMAND ----------
# Inline gate: fail loudly if the demo signal is mis-tuned.
lo, hi = wl.R2_BAND
assert lo <= model_r2 <= hi, f"R2 {model_r2:.3f} outside band {wl.R2_BAND}"

# COMMAND ----------
# Persist as Delta (governed) + CSV on a UC Volume (portable, serverless-safe).
spark.createDataFrame(df).write.mode("overwrite").saveAsTable(DATA_TABLE)
df.to_csv(CSV_PATH, index=False)
print(f"Wrote {DATA_TABLE} and {CSV_PATH}.")
