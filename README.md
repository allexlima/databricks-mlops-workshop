# Databricks MLOps Workshop

A hands-on workshop on the **MLflow model lifecycle** on Databricks — tracking,
the Unity Catalog model registry, validation gates, alias-based promotion, and
composing models into a governed pipeline.

One shared synthetic dataset (a fictional "AnyCompany" commodity buyer) flows
through two model shapes on the *same* lifecycle:

1. a standard ML **forecaster** (scikit-learn) that predicts next-month price, and
2. a custom OR **optimizer** (Pyomo) that turns that prediction into a monthly
   purchase decision — proving MLflow governs even an unusual operations-research
   model, not just framework models.

## Run order (mandatory path)

| # | Notebook | What it does |
|---|----------|--------------|
| 00 | `00_setup` | Create the UC catalog/schema/volume + MLflow experiment |
| 01 | `01_generate_data` | Generate the synthetic dataset (Delta + CSV), with an inline R²-band check |
| 02 | `02_train_forecaster_sklearn` | Train the sklearn forecaster, validate, register + promote `@champion` |
| 03 | `03_register_optimizer_pyomo` | Wrap the Pyomo optimizer as an MLflow PyFunc and register it |
| 04 | `04_end_to_end` | Load both models by `@champion` and run drivers → price → decision |
| 05 | `05_verify` | Four checkpoints proving the path works end to end |

Shared helpers: `_config.py` (`%run` from every notebook — set `CATALOG`/`SCHEMA`
here once) and `workshop_lib.py` (pure data-gen + optimizer logic).

## Optional notebooks (`extra/`)

- `extra/train_forecaster_pytorch` — the same lifecycle with a PyTorch model.
- `extra/serving` — deploy a registered model to a Model Serving endpoint.
- `extra/cleanup` — tear down everything the workshop created.

## Dependencies

- **Serverless (recommended):** each notebook declares its environment as PEP 723
  metadata at the top of the source (ML base + any extras), so there's nothing to
  install — just run.
- **Classic / ML cluster:** PEP 723 is ignored; run this once at the top of each
  notebook (or install on the cluster):
  ```
  %pip install -q -r requirements.txt
  %restart_python
  ```

All model references use the `@champion` alias, never version numbers. Synthetic
data only; fully reproducible from a fixed seed.
