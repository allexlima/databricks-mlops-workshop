# Clonar o repositório

Os notebooks do workshop vivem no GitHub. Para rodá-los na Databricks, você vai
trazer o repositório para dentro do seu workspace como uma **Git folder** (o
recurso antes chamado de Repos). Assim os notebooks ficam versionados e, o mais
importante, o `%run ./_config` e o `import workshop_lib` resolvem os caminhos
relativos corretamente.

!!! note "Conceito"
    Uma **Git folder** é um clone de um repositório Git dentro do workspace da
    Databricks. Ela mantém a estrutura de pastas do repositório, o que faz os
    imports relativos (`import workshop_lib`) e o `%run ./_config` funcionarem do
    mesmo jeito que funcionariam na sua máquina. Se você apenas importar os
    arquivos soltos, esses caminhos quebram.

## Passo a passo

1. Copie a URL do repositório:

    ```text
    https://github.com/allexlima/databricks-mlops-workshop.git
    ```

2. No seu workspace, abra o menu **Workspace** na barra lateral, navegue até a
   pasta onde quer guardar o material (por exemplo, a sua pasta de usuário) e
   clique em **Create › Git folder**.

3. Cole a URL no campo **Git repository URL**. O provedor (**GitHub**) e o nome da
   pasta são preenchidos automaticamente. Mantenha a branch **`main`**, que é onde
   ficam os notebooks.

4. Clique em **Create Git folder**. A Databricks clona o repositório e abre a
   pasta com os notebooks do caminho obrigatório (`00_setup` até `05_verify`) na
   raiz, mais a pasta `extra/` com os labs opcionais.

!!! tip "Curiosidade"
    Como o repositório é público, você não precisa configurar credenciais de Git
    para cloná-lo. Você só vai precisar de um token de acesso (em **Settings ›
    Linked accounts**) se quiser dar `pull` em atualizações futuras ou trabalhar
    com um fork privado seu.

## Como o repositório está organizado

| Caminho | O que é |
|---------|---------|
| `_config.py` | Constantes editáveis (`CATALOG`, `SCHEMA`), nomes derivados, `SEED`, `R2_THRESHOLD` |
| `workshop_lib.py` | Funções compartilhadas e o modelo PyFunc do optimizer |
| `00_setup` … `05_verify` | O caminho obrigatório, na ordem |
| `extra/` | Labs opcionais (PyTorch, serving e cleanup) |

!!! success "Pronto quando…"
    A Git folder aparece no seu workspace e você consegue abrir o `00_setup`. A
    partir daí, siga para os pré-requisitos e ajuste o `_config` para o seu
    catálogo e schema.

Próximo passo: [Pré-requisitos](prerequisites.md) para conferir permissões e
preparar o ambiente antes de rodar o `00_setup`.
