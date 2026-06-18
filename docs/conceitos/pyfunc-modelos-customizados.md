# PyFunc e modelos customizados

No MLflow, cada framework de ML tem seu próprio *flavor*: `mlflow.sklearn`, `mlflow.pytorch`, `mlflow.transformers` e assim por diante. Esses flavors sabem exatamente como serializar, deserializar e servir modelos do framework correspondente. Mas e quando o modelo não pertence a nenhum framework? E quando não existe um `fit()`?

É para isso que existe o `mlflow.pyfunc`.

## O que é o PyFunc

O PyFunc é a interface universal do MLflow. Em vez de depender de um framework específico, ele define um contrato simples: qualquer objeto Python que implemente o método `predict` pode ser registrado, versionado, governado e servido, exatamente como um modelo sklearn ou PyTorch.

!!! note "Conceito"
    `mlflow.pyfunc.PythonModel` é a classe base que você estende para criar um modelo customizado.
    O único método obrigatório é `predict`. O MLflow cuida de tudo o mais: empacotamento,
    registro, versionamento, serving, e integração com Unity Catalog.

Essa universalidade é o que permite ao workshop demonstrar algo contraintuitivo: um **optimizer de pesquisa operacional** (Pyomo + HiGHS) que nunca foi treinado vive no mesmo registry, com o mesmo alias `@champion`, e é carregado pela mesma chamada `mlflow.pyfunc.load_model()` que o forecaster sklearn. Nenhum tratamento especial.

## A assinatura `predict`

Todo PyFunc customizado precisa implementar:

```python
def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
    ...
```

Cada argumento tem um papel preciso:

| Argumento | O que é | Quando usar |
|-----------|---------|-------------|
| `context` | `mlflow.pyfunc.PythonModelContext` que dá acesso aos `artifacts` declarados no `log_model` | Para carregar arquivos extras: pesos, tokenizers, configurações |
| `model_input` | `pd.DataFrame` com os dados de entrada, garantido pelo runtime MLflow | Sempre: é a entrada principal |
| `params` | `dict` opcional com hiperparâmetros de inferência | Para expor opções sem mudar a assinatura (temperatura, limiar, etc.) |

!!! tip "Curiosidade"
    O parâmetro `params=None` foi adicionado na MLflow 2.6 e é **obrigatório na assinatura** nas versões 3.x.
    Mesmo que você não use params, declare-o. Sem ele, o serving pode falhar ao passar configurações extras.

No workshop, o `PurchaseOptimizerModel` em `workshop_lib.py` usa `model_input` para receber as seis variáveis do problema de compra e ignora `context` e `params` porque o solver não precisa de arquivos externos (veja o modelo de otimização em [Otimização com Pyomo](otimizacao-pyomo.md)):

```python
class PurchaseOptimizerModel(mlflow.pyfunc.PythonModel):
    """Wraps the Pyomo optimizer as an MLflow PyFunc so a non-trainable OR model
    lives in the same registry/lifecycle as the ML predictor."""

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        rows = [
            solve_purchase(
                predicted_price=r["predicted_price"],
                holding_cost=r["holding_cost"],
                purchase_cost=r["purchase_cost"],
                demand=r["demand"],
                capacity=r["capacity"],
                budget=r["budget"],
            )
            for _, r in model_input.iterrows()
        ]
        return pd.DataFrame(rows, columns=["purchase_qty", "total_cost", "status"])
```

Cada linha do DataFrame de entrada é um problema de otimização independente. O método retorna um DataFrame com a decisão: quantidade a comprar, custo total, e o status do solver.

## Como registrar: `log_model` no MLflow 3.x

A chamada que empacota e registra o modelo customizado é `mlflow.pyfunc.log_model`. No MLflow 3.x, o parâmetro correto é `name=`, não o antigo `artifact_path=`, que foi depreciado:

```python
with mlflow.start_run(run_name="pyomo_optimizer"):
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=wl.PurchaseOptimizerModel(),
        code_paths=["./workshop_lib.py"],
        pip_requirements=["pyomo>=6.7", "highspy>=1.7", "pandas>=2.0"],
        input_example=example,
        signature=signature,
        registered_model_name=OPTIMIZER_MODEL,
    )
```

!!! warning "Atenção"
    Usar `artifact_path=` em vez de `name=` no MLflow 3.x gera um `FutureWarning` e em versões futuras
    vai falhar. Se você ver esse aviso num notebook, é hora de migrar.

Os parâmetros mais importantes:

**`python_model`**: a instância da sua classe. O MLflow serializa o objeto com `cloudpickle` e o empacota no artefato do run.

**`code_paths`**: lista de arquivos `.py` (ou diretórios) de que o modelo precisa em tempo de execução. No notebook `03_register_optimizer_pyomo.py`, `workshop_lib.py` é declarado aqui porque `PurchaseOptimizerModel.predict` chama `solve_purchase`, que está definida nesse arquivo. Sem `code_paths`, o modelo seria registrado mas falharia ao ser carregado em outro ambiente.

**`pip_requirements`**: dependências Python declaradas explicitamente. O MLflow as inclui no `MLmodel` e as instala automaticamente quando o modelo é servido ou carregado em outro ambiente.

**`signature`**: define o schema das entradas e saídas. Inferida automaticamente com `mlflow.models.infer_signature(example_input, example_output)`. Habilita validação em serving e documentação automática na UI.

**`registered_model_name`**: o nome completo no Unity Catalog: `{catalog}.{schema}.purchase_optimizer`. Declarar aqui registra o modelo diretamente dentro do `with mlflow.start_run()`, sem precisar de uma chamada separada ao `MlflowClient`.

## O mecanismo `context` e `artifacts`

Quando o modelo precisa carregar arquivos em tempo de execução (um arquivo de pesos, um tokenizer, uma tabela de lookup), você os declara em `artifacts` no `log_model` e os acessa via `context.artifacts` dentro do `predict`:

```python
# Ao registrar:
mlflow.pyfunc.log_model(
    name="model",
    python_model=MeuModelo(),
    artifacts={"config": "./config.json", "lookup_table": "./lookup.parquet"},
    ...
)

# Dentro de predict:
def predict(self, context, model_input, params=None):
    config_path = context.artifacts["config"]   # caminho local após download
    # carrega o arquivo...
```

O MLflow faz o upload desses arquivos junto com o artefato do run e os baixa automaticamente quando o modelo é carregado, seja localmente ou num endpoint de serving.

!!! tip "Curiosidade"
    O `context` também expõe `context.model_config` para configurações passadas em tempo de deploy,
    sem precisar re-registrar o modelo. Útil para feature flags ou parâmetros de ambiente.

No optimizer do workshop, o `context` não é usado porque o Pyomo constrói o modelo de otimização do zero a cada chamada. Não há estado pré-computado para carregar.

## Serialização com cloudpickle

Quando você passa `python_model=MeuModelo()`, o MLflow usa `cloudpickle` para serializar o objeto. O `cloudpickle` consegue serializar closures, lambdas, e objetos que o `pickle` padrão não consegue, por isso é a escolha padrão para PyFuncs.

!!! warning "Atenção"
    O cloudpickle serializa o *estado* do objeto no momento do `log_model`, mas **não** serializa o código-fonte das classes.
    Se `PurchaseOptimizerModel` estiver definida no notebook (não num arquivo separado), o modelo vai falhar ao ser
    carregado em outro ambiente, porque o código não estará disponível.

    A solução é sempre declarar as classes em um módulo separado (como `workshop_lib.py`) e incluí-lo via `code_paths`.
    Assim o MLflow empacota o código junto com o artefato e ele estará disponível onde quer que o modelo seja carregado.

## PyFunc vs. flavors built-in: quando usar cada um

| Situação | Recomendação |
|----------|-------------|
| Modelo sklearn, PyTorch, HuggingFace, XGBoost, etc. | Use o flavor nativo (`mlflow.sklearn`, `mlflow.pytorch`, ...) |
| Modelo de qualquer framework que precisa de pré/pós-processamento customizado | PyFunc wrapping o flavor nativo |
| Pipeline multi-etapa (forecaster → optimizer, por exemplo) | PyFunc compondo os dois |
| Solver de otimização, motor de regras, heurística, simulação | PyFunc: é exatamente o caso de uso |
| Modelo sem `fit()` algum | PyFunc: não há outra opção |

O optimizer do workshop é o caso mais puro: um modelo de pesquisa operacional que **nunca foi treinado**. Não há parâmetros aprendidos. O "modelo" é o problema de programação linear e o solver HiGHS. O PyFunc é a única interface que faz sentido, e é suficiente para que esse objeto viva no Unity Catalog com versões, aliases e toda a governança do MLflow.

## Alias `@champion` e acesso version-agnostic

Após o `log_model`, o notebook promove a versão registrada para o alias `@champion`:

```python
client = MlflowClient()
client.set_registered_model_alias(OPTIMIZER_MODEL, "champion", info.registered_model_version)
```

A partir daí, qualquer notebook que precise do optimizer usa o URI com alias, nunca um número de versão fixo:

```python
opt = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
print(opt.predict(example))
```

!!! tip "Curiosidade"
    Usar `@champion` em vez de `@v3` (ou `@latest`, que não existe no Unity Catalog) é a prática correta
    porque re-execuções do notebook criam novas versões. Um número hardcoded quebraria no segundo run.
    O alias é o handle estável que o operator do modelo controla explicitamente: uma decisão de governança,
    não só de conveniência.

Esse padrão é idêntico ao do forecaster sklearn em `02_train_forecaster_sklearn.py`. Os dois modelos (um treinado, outro não) usam exatamente a mesma mecânica de registro e promoção. É o PyFunc que torna isso possível.

## Verificação de viabilidade antes de registrar

O notebook `03_register_optimizer_pyomo.py` não registra cegamente: antes do `log_model`, ele executa o optimizer num exemplo sintético e verifica que a solução é viável:

```python
sample_out = wl.PurchaseOptimizerModel().predict(None, example)
row = sample_out.iloc[0]
assert row["status"] == "optimal", f"optimizer infeasible on example: {row.to_dict()}"
assert 100.0 - 1e-6 <= row["purchase_qty"] <= 200.0 + 1e-6, f"qty out of bounds: {row['purchase_qty']}"
```

Só após passar esses asserts o modelo é registrado e promovido. É o equivalente, para o optimizer, do gate R² ≥ 0.6 do forecaster: ambos exigem evidência de qualidade antes de chegar ao registry.

## Por que PyFunc é a língua franca do MLflow

Todo flavor nativo do MLflow (sklearn, PyTorch, HuggingFace) é implementado *sobre* o PyFunc. Quando você faz `mlflow.sklearn.log_model(...)`, o MLflow salva um wrapper PyFunc que sabe como carregar e chamar o modelo sklearn. A diferença é que para frameworks conhecidos o MLflow gera esse wrapper automaticamente; para modelos customizados, você escreve o wrapper você mesmo.

Isso tem uma consequência importante: **qualquer ferramenta que entende PyFunc entende todos os modelos MLflow**. O Model Serving na Databricks, o `mlflow.evaluate()` e o Unity Catalog interagem com modelos através da interface PyFunc, independentemente do que está por baixo. Registrar o Pyomo optimizer como PyFunc não é uma gambiarra: é o caminho oficial.

---

## Próximos passos

- Veja a implementação completa em [Registrar o optimizer](../lab-3-optimizer/index.md)
- Para entender a lógica do problema de otimização em si (o modelo Pyomo, as restrições, o solver HiGHS): [Otimização com Pyomo](otimizacao-pyomo.md)

Pronto para começar o caminho obrigatório? [Configure o ambiente de trabalho](../setup/workspace.md) e siga a trilha dos labs.
