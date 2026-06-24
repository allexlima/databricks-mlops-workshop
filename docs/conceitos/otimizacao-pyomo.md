# Otimização (pesquisa operacional) com Pyomo e HiGHS

Quando alguém diz "modelo", a primeira imagem costuma ser de dados de treino, uma função de perda e um `fit()`. Mas existe uma família inteira de modelos que **não aprendem**: eles **decidem**. Esta página explica como essa família funciona, como o Pyomo a expressa em Python e por que ela cabe perfeitamente no mesmo ciclo de vida MLflow que qualquer modelo de ML.

---

## Pesquisa operacional: a arte de tomar a melhor decisão possível

**Pesquisa operacional** (PO, ou *operations research*, OR) é o ramo da matemática aplicada que trata de encontrar a solução ótima para problemas de decisão com recursos limitados. Ao contrário do ML, que induz padrões a partir de dados históricos, a PO parte de um modelo matemático explícito do problema e resolve esse modelo de forma exata (ou aproximada, dependendo da complexidade).

O paradigma central é a **otimização matemática**: dado um conjunto de variáveis de decisão, uma função a minimizar (ou maximizar) e um conjunto de restrições, encontre os valores das variáveis que otimizam a função sem violar nenhuma restrição.

**Otimização matemática** é o processo de encontrar o ponto `x*` no espaço viável tal que `f(x*)` seja mínimo (ou máximo). O espaço viável é definido pelas restrições: desigualdades e igualdades que `x` deve satisfazer. Quando tanto `f` quanto as restrições são lineares nas variáveis de decisão, o problema é chamado de **programação linear** (PL ou *linear programming*, LP).

---

## Prever versus decidir

A distinção entre ML e PO é simples, mas importante:

| | ML (forecaster) | PO (optimizer) |
|---|---|---|
| **Pergunta respondida** | "Qual será o preço no próximo mês?" | "Quanto devo comprar este mês?" |
| **Entrada** | Dados históricos de treino | Modelo matemático do problema |
| **Processo** | Ajuste de parâmetros (`fit`) | Resolução de sistema de equações/inequações |
| **Saída** | Predição probabilística | Decisão ótima (determinística) |
| **Melhora com mais dados?** | Sim | Não. Muda-se o modelo, não os dados |
| **Garantia** | Estatística (em média, tende a acertar) | Matemática (solução ótima dentro do modelo) |

**ML e PO se complementam.** Os dois paradigmas se **complementam** no workshop da AnyCompany: o forecaster (ML) prevê o preço do próximo mês, e o optimizer (PO) usa essa previsão como dado de entrada para decidir a quantidade a comprar. Um alimenta o outro, e essa composição é exatamente o que `04_end_to_end.py` monta.

---

## Anatomia de um modelo de otimização

Todo problema de otimização tem três componentes fundamentais:

### 1. Variáveis de decisão

São as incógnitas que o solver vai determinar, as "alavancas" que você controla. No problema da AnyCompany:

- `q`: quantidade a comprar (unidades, real não-negativo)
- `leftover`: excedente sobre a demanda (auxiliar para calcular custo de estoque)

### 2. Função objetivo

É a métrica que você quer minimizar (ou maximizar). Aqui: **minimizar o custo total de compra mais o custo de estoque do excedente**.

```
minimizar:  purchase_cost × q  +  holding_cost × leftover
```

### 3. Restrições

São as fronteiras do espaço viável, ou seja, o que é fisicamente ou logicamente possível:

| Restrição | Descrição | Expressão |
|---|---|---|
| Atender demanda | Comprar ao menos o necessário | `q ≥ demand` |
| Respeitar capacidade | Não exceder o armazém | `q ≤ capacity` |
| Ficar no orçamento | Não ultrapassar o limite financeiro | `purchase_cost × q ≤ budget` |
| Definir excedente | Auxiliar para o custo de estoque | `leftover ≥ q − demand` |

**Viável versus ótima.** Uma solução é **viável** quando satisfaz todas as restrições. Uma solução **ótima** é a melhor dentre todas as viáveis segundo a função objetivo. Se nenhuma combinação de valores satisfaz todas as restrições simultaneamente, o problema é chamado de **infeasible** (sem solução).

---

## Programação linear

O problema da AnyCompany é um caso particular e poderoso: tanto a função objetivo quanto todas as restrições são **lineares** nas variáveis de decisão (nenhum produto entre variáveis, nenhum expoente). Isso torna o problema uma **LP**, com propriedades muito convenientes:

- A solução ótima, se existir, está sempre em um **vértice** do poliedro viável (geometricamente, num "canto" do espaço de soluções).
- Algoritmos como o **simplex** e os de **pontos interiores** resolvem LPs de forma exata e muito eficiente, mesmo com milhões de variáveis.

**A história do simplex.** O método simplex foi proposto por **George Dantzig** em 1947, enquanto trabalhava para as Forças Aéreas dos EUA em problemas de logística militar. Dantzig conta que, ao visitar o matemático John von Neumann com o rascunho do método, von Neumann respondeu em menos de uma hora com a teoria da dualidade LP, que Dantzig não conhecia. O simplex continua sendo um dos algoritmos mais usados na prática mais de 75 anos depois.

---

## Pyomo: otimização matemática em Python

**Pyomo** é uma biblioteca Python de modelagem algébrica para otimização. Em vez de formular o problema em uma linguagem proprietária (AMPL, GAMS), você escreve o modelo em Python puro, o que facilita a integração com pipelines de dados, MLflow e Databricks.

A estrutura básica de um modelo Pyomo:

```python
import pyomo.environ as pyo

m = pyo.ConcreteModel()          # modelo concreto (dados conhecidos agora)
m.q = pyo.Var(domain=pyo.NonNegativeReals)   # variável de decisão
m.leftover = pyo.Var(domain=pyo.NonNegativeReals)

# Restrições
m.meet_demand   = pyo.Constraint(expr=m.q >= demand)
m.cap           = pyo.Constraint(expr=m.q <= capacity)
m.budget        = pyo.Constraint(expr=purchase_cost * m.q <= budget)
m.leftover_def  = pyo.Constraint(expr=m.leftover >= m.q - demand)

# Função objetivo
m.obj = pyo.Objective(
    expr=purchase_cost * m.q + holding_cost * m.leftover,
    sense=pyo.minimize,
)
```

Leia o código como leria matemática: cada linha é uma equação ou inequação do problema. Não há nada implícito. O modelo é exatamente o que está escrito.

**Modelos concretos versus abstratos.** Pyomo oferece dois sabores: `ConcreteModel` (dados embutidos no momento da construção, como aqui) e `AbstractModel` (estrutura separada dos dados, populada depois). Para problemas de decisão mensal com parâmetros variando a cada chamada, `ConcreteModel` é mais direto e legível.

---

## O solver HiGHS via `highspy`

Um modelo Pyomo é apenas uma **representação** do problema: ele não resolve nada sozinho. Você precisa de um **solver** para isso. Existem vários: GLPK, CPLEX, Gurobi, CBC... O workshop usa o **HiGHS**.

**HiGHS** (*High-performance Software for Linear Optimization*) é um solver de código aberto de alto desempenho desenvolvido pela Universidade de Edinburgh. Ele suporta LP, QP e MIP (programação inteira mista), e é notável por várias razões práticas:

- **Puro pip**: o pacote `highspy` empacota o binário HiGHS compilado e o instala como extensão Python. **Nenhum binário externo** precisa ser instalado no sistema operacional. Em ambientes serverless na Databricks, onde você não tem acesso de root, isso é essencial.
- **Interface APPSI**: o Pyomo expõe o HiGHS via a interface APPSI (*Algebraic Programming System Plugin Interface*), mais moderna e eficiente que os adaptadores de linha de comando.
- **Performance**: HiGHS regularmente vence benchmarks contra solvers comerciais em LPs de médio porte.

**HiGHS no OR-Tools e no SciPy.** O HiGHS não é exclusividade do Pyomo. O Google OR-Tools e o `scipy.optimize.linprog` (com `method='highs'`) também o usam internamente como backend LP. Se você já usou `scipy.optimize.linprog` recentemente, provavelmente já rodou o HiGHS sem saber.

---

## `solve_purchase` por dentro

Esta é a função completa em `workshop_lib.py`:

```python
def solve_purchase(predicted_price: float, holding_cost: float, purchase_cost: float,
                   demand: float, capacity: float, budget: float) -> dict:
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers.highs import Highs

    m = pyo.ConcreteModel()
    m.q       = pyo.Var(domain=pyo.NonNegativeReals)
    m.leftover = pyo.Var(domain=pyo.NonNegativeReals)

    m.meet_demand  = pyo.Constraint(expr=m.q >= demand)
    m.capacity     = pyo.Constraint(expr=m.q <= capacity)
    m.budget       = pyo.Constraint(expr=purchase_cost * m.q <= budget)
    m.leftover_def = pyo.Constraint(expr=m.leftover >= m.q - demand)

    m.obj = pyo.Objective(
        expr=purchase_cost * m.q + holding_cost * m.leftover,
        sense=pyo.minimize,
    )

    opt = Highs()
    opt.config.load_solution = False      # não levanta exceção se infeasible
    result = opt.solve(m)

    status = str(result.termination_condition).lower()
    if "optimal" not in status:
        return {"purchase_qty": float("nan"), "total_cost": float("nan"), "status": "infeasible"}

    result.solution_loader.load_vars()    # carrega valores nas variáveis do modelo
    return {
        "purchase_qty": float(pyo.value(m.q)),
        "total_cost":   float(pyo.value(m.obj)),
        "status":       "optimal",
    }
```

Alguns detalhes importantes:

**`opt.config.load_solution = False`**: por padrão, se o solver retornar infeasible ou unbounded, o Pyomo levanta uma exceção ao tentar carregar a solução. Com `load_solution = False`, o controle fica na sua mão: você verifica `termination_condition` e decide o que retornar. Isso permite que o código seja robusto a qualquer combinação de parâmetros que o usuário envie.

**`result.solution_loader.load_vars()`**: só é chamado quando o status é "optimal". Carrega os valores ótimos nas variáveis `m.q` e `m.leftover` para que `pyo.value()` funcione.

**`predicted_price` não entra no solver**: note que `predicted_price` é recebido como parâmetro mas **não é usado diretamente nas restrições ou na função objetivo** do modelo mostrado acima. Ele fica disponível no contexto do `predict()` da `PurchaseOptimizerModel` para possíveis extensões futuras (por exemplo, modelar o risco de variação de preço). Quem de fato entra na função objetivo é `purchase_cost`, que representa o preço de compra do período.

---

## Infeasibilidade: quando não existe solução

Um modelo LP é **infeasible** quando as restrições são contraditórias entre si: não existe nenhum `q` que satisfaça todas ao mesmo tempo. Exemplo clássico:

```
q ≥ 150    (demanda mínima de 150 unidades)
q ≤ 120    (capacidade máxima de 120 unidades)
```

Não existe nenhum valor de `q` que seja simultaneamente ≥ 150 e ≤ 120. O solver detecta isso imediatamente e retorna `termination_condition = infeasible`.

No workshop, isso pode acontecer se o `budget` for muito baixo para atender a `demand` ao `purchase_cost` dado. A função `solve_purchase` retorna `{"status": "infeasible", "purchase_qty": nan, "total_cost": nan}` (um dicionário controlado, sem exceção) e o código chamador pode reagir adequadamente.

!!! warning "Atenção"
    Um modelo infeasible **não é um bug do solver**: é um sinal de que os parâmetros do problema são inconsistentes com o modelo formulado. Na prática, infeasibilidade pode indicar que uma restrição está muito apertada, que os dados de entrada estão fora do intervalo esperado, ou que o modelo precisa ser reformulado para capturar uma flexibilidade que existe no mundo real mas não no modelo.

---

## De solver a MLflow PyFunc

O Pyomo resolve o problema, mas como esse modelo vive no Unity Catalog junto com o forecaster sklearn? A resposta é o `PurchaseOptimizerModel` (definido em `workshop_lib.py`): uma classe que adapta a interface do solver ao contrato PyFunc, recebendo um DataFrame, chamando `solve_purchase` linha a linha e devolvendo um DataFrame com a decisão. O registro e a promoção ao alias `@champion` seguem o mesmo padrão do forecaster sklearn. O detalhe crucial é o `code_paths=["./workshop_lib.py"]`: o solver é empacotado dentro do artefato do modelo no Unity Catalog e estará disponível onde quer que o modelo seja carregado, sem depender do caminho do workspace.

O contrato completo do PyFunc, a assinatura `predict`, o `log_model` no MLflow 3.x e a serialização com cloudpickle estão detalhados em [PyFunc e modelos customizados](pyfunc-modelos-customizados.md).

---

## Por que isso importa no contexto do workshop

A tese central do workshop é: **"o MLflow governa o que você tiver, inclusive um modelo de pesquisa operacional fora do comum"**. O `purchase_optimizer` demonstra isso de forma concreta:

- Não tem `fit()`, não tem dados de treino, não tem R². Ainda assim vive no Unity Catalog com versionamento, alias `@champion`, linhagem e governança idênticos ao forecaster.
- Qualquer pessoa com permissão no catalog pode carregar o modelo pelo URI `models:/{catalog}.{schema}.purchase_optimizer@champion` sem saber nada sobre Pyomo ou HiGHS.
- Em `04_end_to_end.py`, os dois modelos são carregados pelo mesmo mecanismo e compostos em uma cadeia: forecaster prediz o preço → optimizer decide a quantidade. A plataforma não distingue os dois.

---

## Próximos passos

- [PyFunc e modelos customizados](pyfunc-modelos-customizados.md): entenda o contrato PyFunc que torna isso possível.
- [Otimizador Pyomo como PyFunc](../lab-3-optimizer/index.md): execute `03_register_optimizer_pyomo.py` passo a passo.

Pronto para começar o caminho obrigatório? [Configure o ambiente de trabalho](../setup/workspace.md) e siga a trilha dos labs.
