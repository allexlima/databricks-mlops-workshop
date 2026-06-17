# Lab 3 — O otimizador Pyomo como modelo customizado

Neste lab você vai empacotar um modelo de **pesquisa operacional** (Pyomo) que **não treina** como um `mlflow.pyfunc.PythonModel` — para que ele viva no **mesmo registro e ciclo de vida governado** que o modelo de ML do Lab 2. Esse é o grande momento do workshop: provar que o MLflow Model Registry não é exclusividade de quem tem um `fit()`.

Abra `03_register_optimizer_pyomo.py`.

---

## O problema que o otimizador resolve

A cada mês, uma empresa precisa decidir **quanto comprar de uma commodity**. O objetivo é minimizar o custo total — custo de compra × quantidade + custo de estoque sobre o excedente — respeitando três restrições duras:

| Restrição         | Expressão Pyomo                        |
|-------------------|----------------------------------------|
| Atender a demanda | `m.q >= demand`                        |
| Respeitar capacidade | `m.q <= capacity`                   |
| Ficar no orçamento | `purchase_cost * m.q <= budget`       |

O solver é o **HiGHS**, chamado via a interface APPSI do Pyomo (`highspy` — apenas `pip`, nenhum binário externo). Toda a lógica está em `workshop_lib.solve_purchase` e é empacotada pela classe `PurchaseOptimizerModel`:

```python
class PurchaseOptimizerModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        rows = [
            solve_purchase(
                predicted_price=r["predicted_price"], holding_cost=r["holding_cost"],
                purchase_cost=r["purchase_cost"], demand=r["demand"],
                capacity=r["capacity"], budget=r["budget"],
            )
            for _, r in model_input.iterrows()
        ]
        return pd.DataFrame(rows, columns=["purchase_qty", "total_cost", "status"])
```

!!! info "A grande ideia — um ciclo de vida para governar todos"
    `PurchaseOptimizerModel` não tem `fit()`, não tem métricas de treino, não tem hiperparâmetros — é programação matemática pura. Ao envolvê-la como PyFunc, ela entra no Unity Catalog com **versionamento, aliases e governança idênticos** aos do forecaster sklearn do Lab 2. Uma só plataforma, dois modelos completamente diferentes.

---

## Passo 1 — Construir o exemplo e rodar o solver

Antes de logar qualquer coisa, o notebook monta um DataFrame de entrada e chama `.predict()` diretamente. Isso serve para confirmar que o ambiente (solver + biblioteca) está funcional:

```python
example = pd.DataFrame([{
    "predicted_price": 210.0,
    "holding_cost":      1.0,
    "purchase_cost":   200.0,
    "demand":          100.0,
    "capacity":        200.0,
    "budget":        30000.0,
}])

sample_out = wl.PurchaseOptimizerModel().predict(None, example)
```

=== "Entrada"

    | predicted_price | holding_cost | purchase_cost | demand | capacity | budget  |
    |----------------:|-------------:|--------------:|-------:|---------:|--------:|
    | 210.0           | 1.0          | 200.0         | 100.0  | 200.0    | 30 000.0 |

=== "Saída"

    | purchase_qty | total_cost | status  |
    |-------------:|-----------:|:--------|
    | 100.0        | 20 000.0   | optimal |

    O solver escolhe exatamente o valor da demanda (100 unidades): comprar mais adicionaria custo de estoque sem nenhum benefício — o mínimo ditado pelas restrições.

---

## Passo 2 — Assert de viabilidade antes de registrar

Um modelo sem métricas de treino precisa de outro critério de qualidade. O notebook usa um `assert` explícito:

```python
row = sample_out.iloc[0]
assert row["status"] == "optimal", \
    f"optimizer infeasible on example: {row.to_dict()}"
assert 100.0 - 1e-6 <= row["purchase_qty"] <= 200.0 + 1e-6, \
    f"qty out of bounds: {row['purchase_qty']}"
```

**Por quê isso importa:** se o solver estiver ausente, as restrições estiverem mal especificadas ou o ambiente não tiver as dependências corretas, o notebook falha **aqui** — barulhento e antes do registro — em vez de registrar um modelo silenciosamente quebrado.

!!! warning "Pegadinhas reais com o solver"
    - **O solver precisa existir no ambiente de execução.** As dependências `pyomo` e `highspy` são declaradas no cabeçalho PEP 723 do notebook — o Databricks Serverless as instala automaticamente. Elas também são passadas como `pip_requirements` dentro de `log_model` para que o ambiente de serving as re-instale.
    - **PEP 723 não consegue fixar index URLs privadas.** Se o seu workspace usa um proxy PyPI interno, adicione a URL do índice na política do cluster ou em uma célula `%pip install` separada — o PEP 723 não tem mecanismo para isso.
    - **`load_solution = False`** diz ao HiGHS para não lançar exceção em caso de inviabilidade. O wrapper verifica `termination_condition` e retorna `status="infeasible"` de forma limpa em vez de quebrar.

---

## Passo 3 — Logar como PyFunc com dependência de código

```python
signature = mlflow.models.infer_signature(example, sample_out)

with mlflow.start_run(run_name="pyomo_optimizer"):
    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=wl.PurchaseOptimizerModel(),
        code_paths=["./workshop_lib.py"],          # (1) embute o wrapper no artefato
        pip_requirements=["pyomo>=6.7", "highspy>=1.7", "pandas>=2.0"],
        input_example=example,
        signature=signature,
        registered_model_name=OPTIMIZER_MODEL,
    )
```

Dois parâmetros merecem atenção especial:

- **`code_paths=["./workshop_lib.py"]`** — sem isso, `solve_purchase` estaria ausente no momento do carregamento. O MLflow embute o arquivo Python diretamente no artefato logado, garantindo que tudo necessário viaje junto.
- **`pip_requirements`** — declara as dependências de forma explícita para o ambiente de serving. A `signature` inferida (de `example` e `sample_out`) permite que o Model Serving valide entradas sem nenhuma execução de treino.

---

## Passo 4 — Definir o alias `@champion`

```python
client.set_registered_model_alias(
    OPTIMIZER_MODEL, "champion", info.registered_model_version
)
```

Mesmo padrão do forecaster. O Lab 4 carrega **ambos os modelos** via `models:/<nome>@champion` — nunca um número de versão fixo. Isso é o que torna a cadeia ponta a ponta robusta a novas versões.

---

## Passo 5 — Recarregar pelo alias e prever

```python
opt = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
print(opt.predict(example))
```

Se esta célula imprimir um DataFrame com `status=optimal`, o modelo está corretamente empacotado e todas as dependências foram resolvidas a partir do artefato logado.

!!! success "Você deve ver…"
    - A célula do assert de viabilidade conclui **sem erro**.
    - O Unity Catalog exibe um novo modelo registrado `purchase_optimizer` (ou o que `OPTIMIZER_MODEL` resolver) com a versão 1 marcada com o alias `@champion`.
    - A chamada final `opt.predict(example)` retorna um DataFrame de uma linha com `purchase_qty=100.0` e `status=optimal`.

---

!!! note "📸 Espaço reservado para captura de tela"
    *Capture aqui: o modelo `purchase_optimizer` registrado no Unity Catalog. Depois substitua por `![Otimizador registrado](../assets/screenshots/lab-3-optimizer.png)`.*

---

**Próximo:** [Lab 4 — Cadeia ponta a ponta](../lab-4-end-to-end/index.md)
