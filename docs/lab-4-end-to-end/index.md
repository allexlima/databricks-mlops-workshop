# Workflow end-to-end

!!! note "Notebook desta página"
    Esta página acompanha o notebook `04_end_to_end.py`. Abra-o na sua Git folder.

Nas etapas anteriores você treinou e registrou dois modelos completamente diferentes no mesmo ciclo de vida governado pelo MLflow:

- **`price_forecaster`**: um modelo scikit-learn que prevê o preço do próximo mês de uma commodity a partir de drivers econômicos.
- **`purchase_optimizer`**: um modelo de pesquisa operacional (Pyomo), embrulhado como PyFunc, que transforma essa previsão em uma decisão de compra.

Neste lab, você vai **compô-los** em uma única cadeia de decisão e entender o que a governança do Unity Catalog oferece de graça.

![Cadeia ponta a ponta: drivers → forecaster @champion → preço previsto → optimizer @champion → decisão de compra](../assets/diagrams/chain.svg){ width="100%" }

---

## O que é composição de modelos governados

**Composição** é a prática de encadear dois ou mais modelos registrados para resolver um problema que nenhum deles resolveria sozinho. Aqui, o `price_forecaster` responde "qual será o preço?" e o `purchase_optimizer` responde "quanto comprar dado esse preço?". O resultado é um pipeline de decisão mensal rastreável, do dado de entrada ao output final.

A composição de modelos não é novidade. Pipelines em produção sempre combinaram etapas. O que muda aqui é que **cada etapa é um modelo registrado e versionado**: você sabe exatamente qual versão do forecaster gerou o preço previsto que alimentou uma determinada versão do optimizer.

### Por que alias-first torna a composição robusta

Todo o workshop usa o alias `@champion` em vez de números de versão literais. Isso tem consequências diretas neste lab:

| Abordagem | O que acontece num retreino |
|---|---|
| `models:/…/3` (versão fixa) | O código aponta para a versão antiga. É necessário editar o notebook manualmente. |
| `models:/…/latest` | Pega sempre a versão mais recente, mesmo sem validação. Sem gate de qualidade. |
| `models:/…@champion` | Aponta apenas para a versão que passou pelo gate de R² ≥ 0,6 e foi promovida. Nenhuma edição no notebook. |

A cadeia deste lab carrega **ambos os modelos por `@champion`**. Se um retreino acontecer e o novo modelo passar pelo gate, basta promover o alias. A cadeia já usa o modelo mais recente na próxima execução.

**De onde vem o alias `@champion`?** O padrão de alias `@champion` vem do champion-challenger clássico de ML em produção, onde um modelo challenger disputa com o campeão atual. Neste workshop usamos um gate determinístico simples (R² ≥ 0,6), sem challenger, mas o alias carrega a mesma semântica: "versão aprovada para produção". O MLflow 3.x generalizou aliases para qualquer string arbitrária; `@champion` é apenas a mais comum.

---

## O que o Unity Catalog governa automaticamente

Quando você carrega modelos por URI do tipo `models:/{catalog}.{schema}.price_forecaster@champion` com `mlflow.set_registry_uri("databricks-uc")`, o Unity Catalog passa a ser o backend do registry. Isso traz:

- **Lineage automático**: o UC conecta a tabela Delta de entrada, os runs do experimento MLflow e os modelos registrados, sem instrumentação extra.
- **Auditoria de acesso**: quem consultou, retreinou ou promoveu um modelo fica registrado no audit log do UC.
- **Permissões granulares**: você pode controlar quem pode ler ou promover cada modelo registrado, usando as mesmas permissões do Unity Catalog que já se aplicam às suas tabelas.
- **Versionamento imutável**: cada versão registrada é um artefato imutável. Você sempre pode voltar e reproduzir qualquer decisão histórica.

**Lineage sem configuração extra.** O lineage do Unity Catalog usa a mesma infraestrutura que rastreia a linhagem de tabelas Delta. O MLflow 3.x integra essa linhagem de forma nativa quando o registry URI aponta para `databricks-uc`. O grafo aparece automaticamente no Catalog Explorer sem nenhuma configuração adicional.

---

## O que você vai fazer neste lab

O notebook `04_end_to_end.py` percorre quatro passos:

1. **Imports e configuração do registry**: apontar MLflow para o Unity Catalog.
2. **Champion guard**: verificar proativamente se `@champion` existe antes de rodar a cadeia.
3. **Carregar ambos os modelos**: por alias, nunca por número de versão.
4. **Rodar a cadeia**: drivers do mês mais recente → preço previsto → decisão de compra.

As duas subpáginas deste lab cobrem esses passos em detalhe:

- [Carregar os modelos por @champion](carregar-modelos.md): como os modelos são carregados e por que o guard existe.
- [Rodar a cadeia e obter a decisão](cadeia-decisao.md): a execução passo a passo e o lineage no Catalog Explorer.

---

## Pré-requisitos

!!! warning "Atenção"
    Este notebook depende dos dois anteriores:

    - O `02_train_forecaster_sklearn.py` deve ter treinado e promovido `price_forecaster@champion`.
    - O `03_register_optimizer_pyomo.py` deve ter registrado e promovido `purchase_optimizer@champion`.

    Se algum dos dois não foi executado (ou o modelo não passou pelo gate de R² ≥ 0,6), o notebook falha imediatamente com uma mensagem clara, e não com um stack trace genérico do registry.

---

**Próximo passo:** [Carregar os modelos por @champion](carregar-modelos.md)
