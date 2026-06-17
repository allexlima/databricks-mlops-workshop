# Workshop de MLOps no Databricks

Bem-vindo(a)! Neste workshop você vai aprender, na prática, o **ciclo de vida de
modelos com MLflow** no Databricks: rastreamento de experimentos, o registro de modelos
no Unity Catalog, portões de validação, promoção por *alias* e a composição de modelos
em um pipeline governado.

Aqui a ciência de dados é o **veículo**, não o destino. O foco é o ciclo de vida — e
para deixá-lo concreto, seguimos uma empresa fictícia, a **AnyCompany**, que compra uma
matéria-prima (uma *commodity*) todos os meses.

![Ciclo de vida do modelo: Rastrear, Registrar, Validar, Compor, Governar](assets/diagrams/lifecycle.svg){ width="100%" }

## A ideia central

Colocamos **dois modelos bem diferentes no mesmo ciclo de vida governado**:

1. um **forecaster** de ML tradicional (scikit-learn) que prevê o preço do próximo mês; e
2. um **otimizador** de pesquisa operacional (Pyomo) que transforma essa previsão na
   decisão de quanto comprar no mês.

!!! quote "A tese do workshop"
    **O MLflow governa o que você tiver — inclusive um modelo de pesquisa operacional
    fora do comum.** Não se trata apenas de "o MLflow suporta vários frameworks".

## O que você vai construir

Um único conjunto de dados sintético alimenta os dois modelos; ambos são registrados e
governados no Unity Catalog e, no final, compostos em uma cadeia que produz a decisão de
compra do mês:

![Cadeia: drivers → forecaster → preço previsto → otimizador → decisão de compra](assets/diagrams/chain.svg){ width="100%" }

## Para quem é

- **Público:** pessoas usuárias de Databricks em nível básico–intermediário, com pouca
  experiência em MLOps.
- **Duração:** ~3 horas.
- **Computação:** *serverless* (recomendado) ou um cluster clássico com Databricks
  Runtime ML.

!!! info "Como funcionam os labs"
    Cada página de lab corresponde a um notebook e guia você na execução. O caminho
    obrigatório são seis notebooks (`00`–`05`); os labs opcionais acrescentam uma
    variante em PyTorch, *serving* de modelo e a limpeza dos recursos.

[Começar pelos pré-requisitos :material-arrow-right:](setup/prerequisites.md){ .md-button .md-button--primary }
