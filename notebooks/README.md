# Workshop notebooks

Run order (mandatory path): `00_setup` → `01_generate_data` → `02_sklearn_baseline`
→ `04_pyomo_pyfunc` → `05_end_to_end` → `06_verify`.

Optional: `97_optional_pytorch`, `99_optional_serving`, `98_optional_cleanup` (teardown).

## Dependencies
- **Serverless (recommended):** each notebook's deps are declared in its serverless
  Environment and persisted as PEP 723 metadata in the source — nothing to install.
- **Classic / ML cluster:** the PEP 723 environment is ignored; run this at the top
  of each notebook (or install on the cluster):
  ```
  %pip install -q -r ../requirements.txt
  %restart_python
  ```

`workshop_lib.py` holds the pure, reusable logic (data generation, the Pyomo
optimizer, the PyFunc wrapper); the notebooks import it rather than duplicating
logic. Verification is inline asserts in the notebooks plus `06_verify`.

Set the `catalog` and `schema` widgets in `_config` to your own Unity Catalog
target. All model loads use the `@champion` alias, never version numbers.
