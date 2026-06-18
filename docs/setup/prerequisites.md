# Checklist do ambiente

Esta é a última parada antes dos labs. Aqui você confirma, em um só lugar, que os
dois passos anteriores deixaram tudo pronto para rodar o caminho obrigatório, do
Lab 1 ao Lab 5.

!!! note "Conceito"
    Um checklist de ambiente é uma prática simples de MLOps: antes de executar um
    pipeline, você garante que os pré-requisitos de acesso, dados e configuração
    estão satisfeitos. Isso evita falhas no meio do caminho e torna a execução
    reproduzível para qualquer pessoa que repita o workshop.

## O que já deve estar pronto

| Passo | O que ficou pronto | Onde |
|-------|--------------------|------|
| Workspace | Acesso a um workspace com Unity Catalog e permissão para criar um schema | [Workspace Databricks](workspace.md) |
| Repositório | Git folder clonada e `_config` apontando para o seu `CATALOG` e `SCHEMA` | [Clonar e configurar o repositório](clone-repo.md) |
| Setup | O `00_setup` criou o schema, o volume e o experimento MLflow | [Clonar e configurar o repositório](clone-repo.md) |

## Confirme antes de começar

- O `00_setup` terminou imprimindo `Setup complete. Using <catálogo>.<schema>.`.
- Você enxerga o schema e o volume `workshop_files` no Catalog Explorer.
- As constantes `CATALOG` e `SCHEMA` no `_config` apontam para um catálogo e schema onde você tem permissão de escrita.
- O compute escolhido (serverless, recomendado, ou um cluster com Databricks Runtime ML) está ativo no seletor do notebook.

!!! info "📸 Screenshot"
    *Reservado:* o Catalog Explorer mostrando o schema do workshop já criado, com
    o volume `workshop_files` dentro dele (confirma que o `00_setup` funcionou).

!!! warning "Atenção"
    Se o `00_setup` falhou com uma mensagem sobre o catálogo não existir, volte para
    [Clonar e configurar o repositório](clone-repo.md) e ajuste a constante `CATALOG`
    para um catálogo que já exista e ao qual você tenha acesso. Conferir privilégios:
    [Unity Catalog privileges](https://docs.databricks.com/en/data-governance/unity-catalog/manage-privileges/privileges.html)
    (documentação oficial).

!!! success "Pronto quando…"
    Todos os itens acima estão confirmados. O ambiente está preparado e
    reproduzível. Pode seguir para o primeiro lab.

Próximo passo: [Gerar o conjunto de dados](../lab-1-generate-data/index.md)
