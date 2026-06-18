# Carregar os modelos por @champion

O primeiro passo de `04_end_to_end.py` é configurar o registry e carregar os dois modelos registrados. Antes disso, o notebook executa um guard que falha com mensagem clara se algum `@champion` ainda não foi promovido.

---

## Imports e configuração do registry

```python
import mlflow
import pandas as pd
import workshop_lib as wl
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
```

A linha `mlflow.set_registry_uri("databricks-uc")` é o ponto de virada: ela redireciona o MLflow para usar o **Unity Catalog** como backend do model registry. Sem ela, o cliente tentaria o registry legado do workspace e os aliases `@champion` não resolveriam contra os modelos registrados no UC.

!!! note "Conceito"
    O MLflow suporta dois registry backends na Databricks: o **workspace model registry** (legado) e o **Unity Catalog registry** (recomendado). Com `set_registry_uri("databricks-uc")`, todos os `mlflow.pyfunc.load_model(...)` e chamadas ao `MlflowClient` passam a operar no UC. Modelos registrados no UC têm o nome completo com três partes: `{catalog}.{schema}.{model_name}`, por exemplo `main.mlops_workshop.price_forecaster`.

---

## O champion guard

```python
try:
    client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
except Exception:
    raise RuntimeError(
        f"No @champion alias on {FORECASTER_MODEL}. Run 02_train_forecaster_sklearn "
        f"and confirm it passed the R2>={R2_THRESHOLD} gate before running this notebook."
    )
```

Antes de tocar nos dados ou carregar qualquer modelo, o notebook verifica proativamente se o alias `@champion` existe no `price_forecaster`. Se não existir, falha imediatamente com uma mensagem acionável.

### Por que um guard explícito?

Sem o guard, o fluxo seria:

1. Carregar dados ✓
2. Tentar `mlflow.pyfunc.load_model("models:/main.mlops_workshop.price_forecaster@champion")`
3. Erro genérico do registry: `RESOURCE_DOES_NOT_EXIST: Alias champion not found`

Esse erro é difícil de diagnosticar: ele não diz qual lab faltou rodar, nem que o problema é o gate de R². Com o guard no topo do notebook, a mensagem explica exatamente o que falta. Você não precisa abrir o stack trace para saber o próximo passo.

!!! warning "Atenção"
    O guard só verifica o `price_forecaster`. O `purchase_optimizer@champion` é verificado indiretamente quando `mlflow.pyfunc.load_model(...)` é chamado mais adiante. Se o Lab 3 não foi executado, o erro virá nesse ponto, menos elegante, mas ainda identificável pelo nome do modelo no traceback.

!!! tip "Curiosidade"
    Esta é uma forma simples do padrão **fail-fast**: detectar o problema o mais cedo possível no fluxo de execução, na posição onde a mensagem de erro ainda tem contexto suficiente para ser útil. Em produção, esse tipo de verificação antecipada é especialmente valioso em pipelines agendados. Um job que falha na primeira célula com mensagem clara é muito mais fácil de operar do que um que falha na décima com um erro genérico.

---

## Carregando os dois modelos por alias

```python
df = spark.table(DATA_TABLE).toPandas().sort_values("month")
latest = df.iloc[[-1]]

forecaster = mlflow.pyfunc.load_model(f"models:/{FORECASTER_MODEL}@champion")
optimizer  = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
```

Ambos os modelos são carregados via `mlflow.pyfunc.load_model(...)` com a URI no formato `models:/{catalog}.{schema}.{model_name}@champion`. As constantes `FORECASTER_MODEL` e `OPTIMIZER_MODEL` vêm de `_config.py` (via `%run ./_config`) e expandem para:

- `main.mlops_workshop.price_forecaster`
- `main.mlops_workshop.purchase_optimizer`

Isso significa que **nenhum número de versão** aparece no código. Se o forecaster for retreinado amanhã, passar pelo gate de R² ≥ 0,6 e tiver seu `@champion` atualizado, este notebook vai carregar a nova versão automaticamente, sem nenhuma edição.

!!! note "Conceito"
    `mlflow.pyfunc.load_model(...)` é a interface unificada do MLflow para carregar qualquer modelo, independentemente do framework com que foi treinado. Tanto o `price_forecaster` (sklearn) quanto o `purchase_optimizer` (Pyomo PyFunc) retornam um objeto com o mesmo método `.predict(input_df)`. A cadeia funciona porque **o contrato de interface é o mesmo**; o framework por baixo é um detalhe de implementação.

!!! warning "Atenção"
    O alias `@champion` deve ser promovido explicitamente; ele não é automático. O Lab 2 só promove para `@champion` se o modelo passou pelo gate de R² ≥ 0,6. Se você retreinar com dados ruins e o modelo não passar pelo gate, o `@champion` continua apontando para a versão anterior (mais segura). Nunca use `models:/.../latest` em um pipeline de produção: ele pega a versão mais recente registrada, independentemente de qualidade.

---

Próximo passo: [Rodar a cadeia e obter a decisão →](cadeia-decisao.md)
