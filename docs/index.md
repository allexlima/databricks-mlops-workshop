<div class="db-hero" markdown>

# MLOps na Databricks

Aprenda, na prática, o **ciclo de vida de modelos com MLflow** na Databricks — um
conjunto de dados, dois tipos de modelo, um ciclo de vida governado.

<span class="db-badges">
<span class="db-badge db-badge--lava">⏱ ~3 horas</span>
<span class="db-badge">Nível básico–intermediário</span>
<span class="db-badge">Serverless / DBR ML</span>
</span>

</div>

Bem-vindo(a)! Este workshop ensina o **ciclo de vida de modelos de ML com MLflow** —
desde o rastreamento de experimentos até a promoção governada em produção — usando o
Databricks como plataforma. A ciência de dados é o **veículo**, não o destino.

Para deixar esse ciclo concreto, seguimos a **AnyCompany**, uma empresa fictícia que
compra uma matéria-prima (uma *commodity*) todo mês. O problema real é decidir
*quanto comprar*. Para isso, a empresa precisa primeiro *prever o preço* do próximo
mês, e então *otimizar a decisão de compra* com base nessa previsão.

!!! note "Conceito"
    **MLOps** é o conjunto de práticas que leva um modelo de ML de um notebook
    exploratório até um artefato confiável, versionado e governado — que pode ser
    auditado, substituído e monitorado ao longo do tempo. Não é sobre infraestrutura
    sofisticada: é sobre **disciplina de ciclo de vida**. O MLflow é a ferramenta
    que implementa essa disciplina na Databricks.

![Ciclo de vida do modelo: Rastrear → Registrar → Validar → Compor → Governar](assets/diagrams/lifecycle.svg){ width="100%" }

## A tese do workshop

A maioria dos tutoriais de MLflow mostra modelos scikit-learn ou PyTorch — casos
nos quais o fluxo `fit → log → register` é natural. A pergunta interessante é outra:

> **E quando o modelo não tem `fit()`?**

Um solver de pesquisa operacional, por exemplo, não é treinado em dados — ele
*resolve* um problema de otimização matemática a cada chamada. Ainda assim, ele
precisa ser versionado, validado, promovido e composto com outros modelos.

!!! quote ""
    **O MLflow governa o que você tiver — inclusive um modelo de pesquisa operacional
    fora do comum.** Não se trata apenas de "o MLflow suporta vários frameworks".

É isso que este workshop demonstra: dois modelos com naturezas completamente
diferentes vivem no **mesmo ciclo de vida governado**, com os mesmos mecanismos de
registro, alias e lineage no Unity Catalog.

## O que você vai construir

Um único dataset sintético e reprodutível alimenta **dois modelos separados**, ambos
registrados no Unity Catalog e compostos em uma cadeia que produz a decisão de compra
do mês:

![Cadeia de decisão: drivers do mês atual → forecaster @champion → preço previsto → otimizador @champion → decisão de compra](assets/diagrams/chain.svg){ width="100%" }

### Os dois modelos

| Modelo | Tipo | Framework | Registro |
|--------|------|-----------|----------|
| `price_forecaster` | Forecaster de regressão | scikit-learn | `mlflow.sklearn.log_model(name=...)` |
| `purchase_optimizer` | Solver de otimização | Pyomo + HiGHS via PyFunc | `mlflow.pyfunc.log_model(name=..., python_model=...)` |

Ambos são promovidos ao alias **`@champion`** somente se passarem por uma validação
explícita. O forecaster precisa atingir **R² ≥ 0,6** no conjunto de teste (held-out).
O optimizer precisa devolver uma solução **feasible** — o solver HiGHS confirma isso
automaticamente.

!!! tip "Curiosidade"
    O alias `@champion` é a chave que torna o pipeline estável entre re-execuções.
    Cada vez que você re-treina e re-registra, o MLflow cria uma nova versão do
    modelo (v2, v3, …). Referenciar uma versão literal (`models:/…/3`) quebra na
    próxima re-execução. Referenciar `@champion` nunca quebra — você simplesmente
    move o alias para a nova versão aprovada.

### O ciclo de vida em cinco etapas

1. **Track** — cada execução de treinamento (ou configuração do solver) é registrada
   como um MLflow *run*: parâmetros, métricas, artefatos.
2. **Register** — o modelo aprovado é promovido ao Unity Catalog como um artefato
   versionado e governado.
3. **Validate** — um portão explícito (R² ≥ 0,6 para o forecaster; feasibility para
   o optimizer) precede qualquer promoção ao alias `@champion`.
4. **Compose** — os dois modelos são carregados pelo alias e compostos em uma cadeia:
   `drivers → forecaster → preço → optimizer → decisão`.
5. **Govern** — Unity Catalog mantém o lineage, a auditoria e o controle de acesso
   de ambos os modelos em um único lugar.

!!! note "Conceito"
    **PyFunc** (abreviação de *Python Function*) é a interface genérica do MLflow para
    modelos que não têm um flavor nativo — qualquer classe Python que implemente
    `predict(self, context, model_input, params=None)` pode ser registrada, versionada
    e servida exatamente como um modelo scikit-learn. É o mecanismo que torna possível
    governar o solver Pyomo no mesmo ciclo de vida. Veja mais em
    [Conceitos: PyFunc e modelos customizados](conceitos/pyfunc-modelos-customizados.md).

## Conceitos de apoio

Dois tópicos conceituais estão disponíveis para leitura antes ou durante os labs —
úteis se você não tem familiaridade com PyFunc ou com otimização via Pyomo:

- [PyFunc e modelos customizados](conceitos/pyfunc-modelos-customizados.md) — como o
  MLflow empacota qualquer objeto Python como modelo registrável e servível.
- [Otimização com Pyomo](conceitos/otimizacao-pyomo.md) — o que é um modelo de
  programação linear, como o Pyomo formula o problema de compra e por que o solver
  HiGHS é uma boa escolha sem dependências de sistema.

## A trilha do workshop

<div class="grid cards" markdown>

-   :material-database-cog:{ .lg } __Lab 1 — Gerar o conjunto de dados__

    ---

    Um dataset sintético, reprodutível e governado — a base de todo o ciclo de vida.

    [:octicons-arrow-right-24: Começar](lab-1-generate-data/index.md)

-   :material-chart-line:{ .lg } __Lab 2 — Forecaster (sklearn)__

    ---

    Treinar, rastrear e promover a `@champion` com um portão de validação.

    [:octicons-arrow-right-24: Abrir](lab-2-forecaster/index.md)

-   :material-cog-sync:{ .lg } __Lab 3 — Otimizador Pyomo__

    ---

    Um modelo de pesquisa operacional (sem `fit()`) governado como PyFunc — o destaque.

    [:octicons-arrow-right-24: Abrir](lab-3-optimizer/index.md)

-   :material-link-variant:{ .lg } __Lab 4 — Cadeia ponta a ponta__

    ---

    Compor os dois modelos por alias e ver o lineage no Unity Catalog.

    [:octicons-arrow-right-24: Abrir](lab-4-end-to-end/index.md)

-   :material-check-decagram:{ .lg } __Lab 5 — Verificação__

    ---

    Quatro checkpoints que provam que o caminho inteiro funcionou.

    [:octicons-arrow-right-24: Abrir](lab-5-verify/index.md)

-   :material-flask-outline:{ .lg } __Labs opcionais__

    ---

    PyTorch, Model Serving e a limpeza dos recursos.

    [:octicons-arrow-right-24: Abrir](optional/index.md)

</div>

## Para quem é

- **Público:** pessoas usuárias de Databricks em nível básico–intermediário, com pouca
  experiência em MLOps.
- **Duração:** ~3 horas.
- **Computação:** *serverless* (recomendado) ou um cluster clássico com Databricks
  Runtime ML.

!!! info "Como funcionam os labs"
    Cada página de lab corresponde a um notebook e guia você na execução. O caminho
    obrigatório são seis notebooks (`00`–`05`); os labs opcionais acrescentam uma
    variante em PyTorch, *serving* de modelo e a limpeza dos recursos. Todos os
    notebooks usam um dataset sintético com semente fixa — cada re-execução produz
    os mesmos resultados.

!!! warning "Atenção"
    O catálogo Unity Catalog (`main` por padrão) precisa existir antes de executar
    o `00_setup.py`. O notebook cria o *schema* e o *volume* automaticamente, mas
    não o catálogo. Se o catálogo não existir, você verá uma mensagem clara pedindo
    para criá-lo primeiro.

[Começar pelos pré-requisitos :material-arrow-right:](setup/prerequisites.md){ .md-button .md-button--primary }
