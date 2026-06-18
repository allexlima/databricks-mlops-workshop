# Workspace Databricks

Para acompanhar este workshop você precisa de um workspace Databricks com **Unity Catalog** habilitado. Há dois caminhos, escolha o que se encaixa melhor na sua situação.

---

## Opção A: seu próprio workspace

Se sua empresa já usa Databricks, você provavelmente pode usar o workspace corporativo. Antes de começar, confirme dois pontos:

**Unity Catalog está habilitado?**
Abra o menu lateral e verifique se existe a seção *Catalog* com a árvore de catálogos/schemas/tabelas. Se você só vê a visão antiga de Data, o workspace ainda não migrou para Unity Catalog, fale com o seu administrador Databricks.

**Você tem as permissões certas?**
O workshop usa um catálogo existente (padrão: `main`) e cria um schema dentro dele. Você vai precisar das seguintes permissões no Unity Catalog:

| Nível | Privilege | Para que serve |
|---|---|---|
| Catálogo | `USE CATALOG` | Acessar o catálogo |
| Catálogo | `CREATE SCHEMA` | Criar o schema do workshop |
| Schema | `CREATE TABLE` | Salvar os dados gerados |
| Schema | `CREATE MODEL` | Registrar os modelos no Unity Catalog |
| Schema | `CREATE VOLUME` | Criar o volume para armazenar arquivos |

!!! warning "Atenção"
    Se o catálogo `main` não for acessível para você (comum em workspaces corporativos com catálogos por time), não se preocupe. Você vai ajustar a constante `CATALOG` no arquivo `_config.py` antes de rodar qualquer notebook, basta trocar `"main"` pelo nome do catálogo em que você tem essas permissões.

!!! note "Conceito"
    **Unity Catalog** é a camada de governança centralizada da Databricks. Diferentemente do Hive Metastore legado (por workspace), o Unity Catalog é compartilhado entre workspaces e gerencia dados, modelos e volumes numa hierarquia de três níveis: `catalog.schema.objeto`. Neste workshop, os dois modelos treinados ficam registrados lá: `{catalog}.{schema}.price_forecaster` e `{catalog}.{schema}.purchase_optimizer`.

    Consulte a [documentação oficial de privileges do Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/manage-privileges/privileges.html) se precisar pedir as permissões ao seu administrador.

!!! info "📸 Screenshot"
    *Reservado:* tela inicial do workspace após login, com o menu lateral mostrando a seção *Catalog* e a árvore de catálogos do Unity Catalog.

---

## Opção B: Databricks Free Edition

Se você está fazendo o workshop por conta própria, ou simplesmente quer um ambiente limpo e isolado, a **Databricks Free Edition** é a escolha ideal.

A Free Edition é uma conta gratuita da Databricks que já vem com:

- **Unity Catalog** habilitado e configurado por padrão
- **Serverless compute** incluído, sem precisar criar nem configurar cluster
- Um catálogo padrão pronto para usar

Para criar sua conta, acesse:
[**Databricks Free Edition**: databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition)

O cadastro leva alguns minutos. Após confirmar o e-mail, você já tem acesso a um workspace funcional.

!!! info "📸 Screenshot"
    *Reservado:* página de cadastro da Free Edition, com o formulário de criação de conta.

!!! tip "Curiosidade"
    O Unity Catalog foi lançado em 2022 e se tornou o padrão de governança da Databricks. A Free Edition o adota por padrão justamente porque ele é o caminho recomendado, inclusive para quem está começando. Workspaces antigos criados antes de 2023 podem ainda estar no Hive Metastore legado; a migração é possível mas precisa de permissões de administrador.

---

## Uma nota sobre compute

Este workshop é pensado para rodar em **serverless compute**: a opção mais simples, sem precisar criar nem gerenciar clusters. Cada notebook já declara o ambiente de execução que precisa; você só abre e roda.

Se o seu workspace não tiver serverless disponível, há um caminho alternativo com clusters clássicos, o próximo passo explica os dois casos.

!!! note "Conceito"
    **Serverless compute** na Databricks significa que a infraestrutura de execução é gerenciada automaticamente pela plataforma, sem escolher tipo de instância, sem aguardar cluster subir, sem pagar por tempo ocioso. Você paga apenas pelo tempo de computação efetivamente usado. Consulte a [documentação oficial de serverless](https://docs.databricks.com/en/compute/serverless/index.html) para mais detalhes.

---

Próximo passo: [Clonar e configurar o repositório](clone-repo.md)
