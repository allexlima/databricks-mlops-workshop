# Workshop notebooks

Run order (mandatory path): `00_setup` → `01_generate_data` → `02_sklearn_baseline`
→ `04_pyomo_pyfunc` → `05_end_to_end` → `06_verify`.

Optional: `97_optional_pytorch`, `99_optional_serving`, `98_optional_cleanup` (teardown).

`workshop_lib.py` holds the pure, reusable logic (data generation, the Pyomo
optimizer, the PyFunc wrapper); the notebooks import it rather than duplicating
logic. Verification is inline asserts in the notebooks plus `06_verify`.

Set the `catalog` and `schema` widgets in `_config` to your own Unity Catalog
target. All model loads use the `@champion` alias, never version numbers.
