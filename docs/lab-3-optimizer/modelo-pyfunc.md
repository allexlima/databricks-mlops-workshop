# Empacotar o otimizador como PyFunc

Esta página acompanha `03_register_optimizer_pyomo.py` e cobre os dois primeiros grandes passos do notebook: **formular e resolver o problema** com Pyomo, e **empacotar o solver** como um `mlflow.pyfunc.PythonModel`.

---

## A formulação Pyomo dentro de `solve_purchase`

Toda a lógica matemática vive em `workshop_lib.solve_purchase`. A função recebe os parâmetros de um mês específico e retorna a decisão ótima de compra. Veja a formulação completa:

```python
def solve_purchase(predicted_price, holding_cost, purchase_cost,
                   demand, capacity, budget) -> dict:
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers.highs import Highs

    m = pyo.ConcreteModel()

    # Variáveis de decisão
    m.q        = pyo.Var(domain=pyo.NonNegativeReals)  # quantidade a comprar
    m.leftover = pyo.Var(domain=pyo.NonNegativeReals)  # excedente (estoque)

    # Restrições duras
    m.meet_demand   = pyo.Constraint(expr=m.q >= demand)
    m.capacity      = pyo.Constraint(expr=m.q <= capacity)
    m.budget        = pyo.Constraint(expr=purchase_cost * m.q <= budget)
    m.leftover_def  = pyo.Constraint(expr=m.leftover >= m.q - demand)

    # Função objetivo: minimizar custo total
    m.obj = pyo.Objective(
        expr=purchase_cost * m.q + holding_cost * m.leftover,
        sense=pyo.minimize
    )

    opt = Highs()
    opt.config.load_solution = False   # não lança exceção em caso de inviabilidade
    result = opt.solve(m)
    ...
```

### O que cada peça faz

| Elemento               | Papel na formulação                                                         |
|------------------------|-----------------------------------------------------------------------------|
| `m.q`                  | Quantidade a comprar (variável de decisão principal, ≥ 0)                   |
| `m.leftover`           | Estoque gerado se `q > demand`; captura o custo de holding                 |
| `m.meet_demand`        | Restrição hard: não há opção de falta (`q ≥ demand`)                        |
| `m.capacity`           | Restrição hard: limite físico de armazenagem (`q ≤ capacity`)               |
| `m.budget`             | Restrição hard: orçamento disponível (`purchase_cost × q ≤ budget`)         |
| `m.leftover_def`       | Vincula `leftover` à decisão de compra: `leftover ≥ q − demand`             |
| `m.obj`                | Minimizar `purchase_cost × q + holding_cost × leftover`                     |

!!! note "Conceito"
    **Por que modelar `leftover` explicitamente?** Se o modelo comprasse exatamente `demand`, o custo de holding seria sempre zero e `leftover` seria desnecessário. Mas se comprar mais do que a demanda for preferível (p.ex. preço hoje mais barato que amanhã), o otimizador vai considerar isso e o `holding_cost` encarece a folga. A variável `leftover` e a restrição `m.leftover_def` capturam esse trade-off de forma linear. Sem ela, o custo de holding não entraria na função objetivo.

### HiGHS via APPSI: por que essa combinação

O Pyomo oferece várias interfaces de solver. A **APPSI** (Auto-Persistent Pyomo Solver Interface) é a mais moderna: ela mantém o solver em memória entre chamadas consecutivas, evitando o overhead de serializar/desserializar o modelo a cada solve. Para o workshop, que resolve um modelo por linha de DataFrame, isso simplifica o código e é performático o suficiente.

O **HiGHS** foi escolhido por ser:

- **Pure-pip**: `pip install highspy` é tudo que precisa. Nenhum binário externo, nenhuma variável de ambiente, nenhuma licença.
- **Performático**: solver de PL e MIP de nível profissional, competitivo com opções comerciais em instâncias de médio porte.
- **Serverless-friendly**: sem dependências do sistema operacional, roda no Serverless da Databricks sem configuração adicional.

```python
opt = Highs()
opt.config.load_solution = False  # crítico: ver abaixo
result = opt.solve(m)
```

!!! warning "Atenção"
    **`load_solution = False` é fundamental.** Por padrão, o HiGHS via APPSI lança uma exceção quando o problema é inviável (p.ex. `budget` tão pequeno que nem a demanda mínima cabe no orçamento). Com `load_solution = False`, o solver retorna normalmente e você verifica `result.termination_condition` para decidir o que fazer. O `solve_purchase` checa a string de status e retorna `{"status": "infeasible", "purchase_qty": NaN, "total_cost": NaN}` de forma limpa, sem stack trace e sem colapso do pipeline.

### Infeasibilidade tratada de forma explícita

```python
status = str(result.termination_condition).lower()
if "optimal" not in status:
    return {"purchase_qty": float("nan"),
            "total_cost": float("nan"),
            "status": "infeasible"}
result.solution_loader.load_vars()
return {
    "purchase_qty": float(pyo.value(m.q)),
    "total_cost":   float(pyo.value(m.obj)),
    "status":       "optimal",
}
```

Problemas inviáveis são realidade operacional: um mês com budget muito apertado, uma capacidade de armazenagem abaixo da demanda contratada. O modelo não esconde isso. Retorna `NaN` e `status="infeasible"` para que o chamador tome a decisão correta.

---

## O wrapper PyFunc: `PurchaseOptimizerModel`

Para que o MLflow possa serializar, registrar, versionar e servir o otimizador como qualquer outro modelo, ele precisa implementar a interface **`mlflow.pyfunc.PythonModel`**:

```python
class PurchaseOptimizerModel(mlflow.pyfunc.PythonModel):

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

### Por que `predict(self, context, model_input, params=None)`

Essa assinatura é o contrato do MLflow 3.x para qualquer PyFunc customizado. Três pontos merecem atenção:

| Parâmetro      | O que é                                                                          |
|----------------|----------------------------------------------------------------------------------|
| `context`      | Objeto MLflow com acesso a artefatos logados (não usado aqui; o solver é stateless) |
| `model_input`  | DataFrame de entrada; o MLflow garante que chegue nesse tipo após o log com `signature` |
| `params`       | Parâmetros opcionais de inferência (não usados aqui, mas a assinatura é obrigatória) |

!!! note "Conceito"
    **Por que um DataFrame de entrada, não escalares?** O PyFunc é uma interface de *batch*: ele aceita múltiplos exemplos de uma vez. Aqui, cada linha do DataFrame representa um mês de decisão independente. O `predict` itera pelas linhas e chama `solve_purchase` para cada uma, o que torna o otimizador compatível com Model Serving (que envia batches) e com o Lab 4 (que pode compor várias decisões de uma vez).

!!! tip "Curiosidade"
    O `PurchaseOptimizerModel` não tem `__init__` nem estado interno. O solver Pyomo é instanciado dentro de `solve_purchase` a cada chamada. Essa escolha é intencional: evita problemas de serialização (o MLflow vai fazer pickle da instância) e garante que cada solve começa de um modelo limpo. Em produção, para alta frequência de chamadas, você poderia manter o solver em memória no `__init__` e usar a persistência da APPSI.

---

## Rodando o exemplo antes de registrar

O notebook constrói um exemplo canônico e chama `.predict()` diretamente, antes de qualquer `log_model`:

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
print(sample_out)
```

=== "Entrada"

    | predicted_price | holding_cost | purchase_cost | demand | capacity | budget    |
    |----------------:|-------------:|--------------:|-------:|---------:|----------:|
    | 210.0           | 1.0          | 200.0         | 100.0  | 200.0    | 30 000.0  |

=== "Saída"

    | purchase_qty | total_cost | status  |
    |-------------:|-----------:|:--------|
    | 100.0        | 20 000.0   | optimal |

    O solver escolhe exatamente `demand = 100` unidades: comprar mais adicionaria custo de holding sem benefício. O mínimo viável é o ótimo.

Esse passo tem duas funções: confirma que o ambiente está correto (Pyomo + highspy instalados e funcionais) e gera o `sample_out` que será usado para inferir a `signature` do modelo.

---

**Próximo passo:** [Registrar e promover a `@champion`](registrar-promover.md)
