# Limpeza dos recursos

Este notebook remove todos os recursos criados pelo workshop: endpoint de serving, modelos registrados, experimento MLflow, tabela Delta, Volume CSV e, por padrão, o schema inteiro.

Abra `extra/cleanup.py`.

!!! warning "Atenção"
    Este notebook **deleta permanentemente** todos os recursos criados pelo workshop. A ação não pode ser desfeita. Leia os dois flags de controle antes de executar qualquer célula.

---

## Dois flags de controle: leia antes de rodar

```python
CONFIRM = True       # 👉 defina False para abortar antes de qualquer deleção
DROP_SCHEMA = True   # 👉 defina False para manter o schema
```

O `assert CONFIRM` é a primeira instrução executável do notebook. Se você mudar para `False`, toda a execução para ali, sem tocar em nenhum recurso.

| Flag | Padrão | Efeito |
|---|---|---|
| `CONFIRM` | `True` | `False` aborta imediatamente via `assert` (nada é tocado). |
| `DROP_SCHEMA` | `True` | `False` preserva o schema; remove apenas modelos, tabela e volume. |

---

## O que cada passo apaga

Cada passo de deleção é independente e protegido por `try/except`: se o lab de serving foi pulado e o endpoint não existe, o notebook continua normalmente para os demais recursos.

```python
# 1. Endpoint de serving
get_deploy_client("databricks").delete_endpoint(SERVING_ENDPOINT)

# 2. Modelos registrados (com todas as versões)
for name in (FORECASTER_MODEL, OPTIMIZER_MODEL):
    client.delete_registered_model(name)

# 3. Experimento MLflow e todas as runs
mlflow.delete_experiment(exp.experiment_id)

# 4. Tabela Delta e Volume CSV
spark.sql(f"DROP TABLE IF EXISTS {DATA_TABLE}")
spark.sql(f"DROP VOLUME IF EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# 5. Schema (se DROP_SCHEMA = True)
spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
```

!!! warning "Atenção ao schema compartilhado"
    Se o catalog ou schema for usado por outros participantes, outros workshops, ou qualquer outra carga de trabalho, **defina `DROP_SCHEMA = False`** antes de executar. O `DROP SCHEMA ... CASCADE` remove tudo dentro do schema sem confirmação adicional do Databricks, inclusive objetos que não foram criados por este workshop.

!!! note "Conceito"
    O `DROP SCHEMA ... CASCADE` é um comando SQL do Unity Catalog que apaga recursivamente todos os objetos dentro do schema: tabelas, volumes, modelos registrados, views, funções. É a forma mais eficiente de fazer uma limpeza completa, mas exige que o schema seja exclusivo do workshop antes de usar.

---

## Saída esperada

A execução bem-sucedida produz uma linha por recurso removido:

- `Deleted serving endpoint.` (ou `No serving endpoint to delete (...)` se o lab de serving foi pulado)
- `Deleted registered model <nome>.` para `FORECASTER_MODEL` e `OPTIMIZER_MODEL`
- `Deleted experiment <caminho>.`
- `Dropped <DATA_TABLE> and volume <VOLUME>.`
- `Dropped schema <CATALOG>.<SCHEMA>.` (apenas se `DROP_SCHEMA = True`)
- `Cleanup complete.`

!!! success "Pronto quando..."
    - A última linha impressa é `Cleanup complete.`
    - O Unity Catalog não lista mais os modelos `price_forecaster` e `purchase_optimizer`.
    - O experimento MLflow não aparece mais na UI.

---

!!! success "Fim do workshop"
    Você percorreu o ciclo de vida completo do MLflow na Databricks:

    - **Track**: experimentos e métricas logados automaticamente a cada run
    - **Register**: dois modelos muito diferentes (`price_forecaster` sklearn e `purchase_optimizer` Pyomo PyFunc) no mesmo Unity Catalog
    - **Validate**: validation gate R² >= 0.6 antes de qualquer promoção
    - **Compose**: cadeia de decisão carregando ambos os modelos pelo alias `@champion`
    - **Govern**: versionamento, aliases e linhagem gerenciados pelo Unity Catalog

    A tese ficou demonstrada: **o MLflow governa o que você tiver**, inclusive um modelo de pesquisa operacional fora do comum. Para revisitar qualquer parte, use a navegação lateral.

    [Voltar ao início](../index.md)
