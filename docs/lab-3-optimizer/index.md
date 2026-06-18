# Otimizador Pyomo (visão geral)

Este é o lab central do workshop. Aqui a tese do curso se torna concreta: o MLflow governa o que você tiver, inclusive um modelo de **pesquisa operacional** que nunca viu uma função `fit()`.

Abra `03_register_optimizer_pyomo.py`.

---

## O problema de negócio da AnyCompany

Todo mês, o time de compras da AnyCompany enfrenta a mesma pergunta: **quanto comprar de matéria-prima commodity?**

A resposta depende de variáveis que mudam a cada período:

- O preço que o forecaster previu para o próximo mês.
- O custo de manter estoque parado (custo de holding).
- A demanda mínima que precisa ser atendida.
- A capacidade máxima de armazenagem.
- O orçamento disponível para o período.

Comprar menos do que a demanda é inviável. Comprar mais do que a capacidade ou do que o orçamento permite também. Dentro desses limites, o objetivo é **minimizar custo**: custo de compra × quantidade + custo de holding sobre o excedente.

| Restrição              | O que representa                                           |
|------------------------|------------------------------------------------------------|
| `q >= demand`          | Atender a demanda mínima (não há opção de falta)           |
| `q <= capacity`        | Respeitar o limite físico de armazenamento                 |
| `purchase_cost × q <= budget` | Não estourar o orçamento do período               |

Esse é um problema de **programação linear** clássico, uma linha de pesquisa operacional com décadas de história. Não há dados de treino, não há pesos aprendidos, não há gradiente: há uma formulação matemática e um solver que encontra a solução ótima.

!!! note "Conceito"
    **Programação linear (PL)** é uma técnica de otimização que encontra o mínimo (ou máximo) de uma função objetivo linear sujeita a restrições lineares. A solução existe e é única se o problema for viável e limitado. A AnyCompany tem exatamente esse perfil: função de custo linear, restrições de capacidade/demanda/orçamento lineares.

---

## Por que um modelo de OR, não mais ML?

O forecaster do Lab 2 *prevê* o preço: dado um conjunto de drivers econômicos, ele estima quanto a commodity vai custar no próximo mês. Mas previsão não é decisão.

Para decidir **quanto comprar**, você precisa de um modelo de *otimização*: one that takes the predicted price as input and finds the quantity that minimizes total cost respecting hard constraints. Esses são problemas fundamentalmente diferentes: o ML generaliza a partir de dados históricos; o OR resolve uma formulação matemática explícita.

Na cadeia ponta a ponta do workshop, os dois modelos trabalham juntos:

```
drivers do mês atual
    → forecaster @champion
        → preço previsto
            → optimizer @champion
                → decisão de compra
```

!!! tip "Curiosidade"
    HiGHS é um solver de programação linear e inteira de código aberto desenvolvido na Universidade de Edinburgh. Em benchmarks independentes, ele rivaliza com solvers comerciais como Gurobi e CPLEX em instâncias de médio porte. É 100% gratuito, 100% `pip install`. Neste workshop, ele é chamado via a interface APPSI do Pyomo (`appsi_highs`), que oferece uma API de alto nível sem nenhum binário externo.

---

## A tese: um ciclo de vida para governar todos

O Lab 2 registrou um modelo sklearn com `mlflow.sklearn.log_model`. O registro e o versionamento foram automáticos, o alias `@champion` foi definido, e o Lab 4 vai carregar esse modelo pelo alias.

Agora, o Lab 3 faz **exatamente o mesmo** com um modelo que:

- Não tem `fit()`.
- Não tem métricas de treino.
- Não tem hiperparâmetros para tunar.
- É resolvido por um solver externo, não por backpropagation.

A chave é o **MLflow PyFunc**: uma interface genérica que aceita qualquer objeto Python com um método `predict(self, context, model_input, params=None)`. Ao empacotar o otimizador Pyomo como `PurchaseOptimizerModel(mlflow.pyfunc.PythonModel)`, ele entra no Unity Catalog com versionamento, aliases e governança idênticos aos do forecaster. Uma só plataforma, dois modelos completamente diferentes.

!!! warning "Atenção"
    Um modelo sem métricas de treino precisa de outro critério de qualidade antes do registro. Neste lab, usamos um **assert de viabilidade**: o solver deve retornar `status="optimal"` em um exemplo canônico. Se o ambiente estiver errado (solver ausente, dependências faltando, restrições mal especificadas), o notebook falha aqui, de forma barulhenta e antes do registro, em vez de registrar um modelo silenciosamente quebrado.

---

## O que você vai fazer neste lab

Este lab está dividido em duas páginas:

1. **[Empacotar o otimizador como PyFunc](modelo-pyfunc.md)**: como o Pyomo formula e resolve o problema, e como `PurchaseOptimizerModel` empacota isso como um `mlflow.pyfunc.PythonModel` com `predict()`.

2. **[Registrar e promover a `@champion`](registrar-promover.md)**: fazer o log do modelo no Unity Catalog com `code_paths`, definir o alias `@champion`, e recarregar pelo alias para confirmar que tudo funcionou.

### Leitura de apoio

Antes ou durante o lab, as páginas de conceitos aprofundam os dois pilares técnicos:

- [Otimização com Pyomo](../conceitos/otimizacao-pyomo.md): formulação de modelos PL, APPSI, HiGHS, infeasibilidade.
- [PyFunc: modelos customizados](../conceitos/pyfunc-modelos-customizados.md): a interface genérica do MLflow para qualquer objeto Python com `predict()`.

---

**Próximo passo:** [Empacotar o otimizador como PyFunc](modelo-pyfunc.md)
