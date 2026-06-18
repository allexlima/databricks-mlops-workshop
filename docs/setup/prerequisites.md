# Pré-requisitos

Antes de executar o primeiro notebook, confirme que o ambiente está pronto. Esta página explica **o que você precisa, por quê, e como configurar**: não apenas o checklist, mas o raciocínio por trás de cada item.

---

## 1. Workspace Databricks com Unity Catalog

Todo o workshop gira em torno do **Unity Catalog** (UC), o sistema de governança centralizado da Databricks que gerencia tabelas, volumes, modelos e permissões sob um namespace unificado de três níveis.

!!! note "Conceito: o namespace de três níveis"
    O Unity Catalog organiza todos os dados e objetos na forma `catalog.schema.objeto`:

    | Nível | Responsabilidade | Exemplo |
    |---|---|---|
    | **Catalog** | Isolamento organizacional (unidade de negócio, ambiente) | `main` |
    | **Schema** | Agrupamento lógico de objetos relacionados | `mlops_workshop` |
    | **Objeto** | Tabela, Volume, modelo MLflow, função | `price_forecaster` |

    Ao fazer `mlflow.set_registry_uri("databricks-uc")`, você instrui o MLflow a registrar modelos nesse namespace, em vez do registry legado por workspace. Um modelo registrado fica em `catalog.schema.nome_do_modelo`, e você pode promovê-lo com um alias como `@champion` que persiste entre re-runs sem depender de número de versão.

!!! tip "Curiosidade: por que o UC mudou o jogo"
    Antes do Unity Catalog, cada workspace Databricks tinha seu próprio registry de modelos isolado. Mover um modelo entre ambientes exigia exportação manual. Com o UC, o mesmo modelo vive em um namespace centralizado e pode ser acessado por qualquer workspace conectado ao mesmo metastore. Para este workshop, isso significa que o `@champion` registrado em `02_train_forecaster_sklearn.py` está disponível automaticamente no `04_end_to_end.py`, sem nenhuma cópia.

Verifique se o seu workspace tem Unity Catalog habilitado: na barra lateral, você deve ver a opção **Catalog** (ícone de tabela). Se não aparecer, entre em contato com o administrador do workspace.

---

## 2. Permissões no Unity Catalog

O notebook `00_setup.py` cria o **schema** e o **Volume** dentro de um catálogo existente, mas **não cria o catálogo**. Criar catálogos é uma operação privilegiada que depende da configuração de *managed location* (local de armazenamento gerenciado) no metastore, e é responsabilidade do administrador.

Você precisa das seguintes permissões no catálogo escolhido:

| Permissão | Para quê |
|---|---|
| `USE CATALOG` | Enxergar o catálogo e seus schemas |
| `CREATE SCHEMA` | Criar `mlops_workshop` dentro do catálogo |
| `CREATE TABLE` | Salvar a tabela `commodity_monthly` no schema |
| `CREATE VOLUME` | Criar o volume `workshop_files` para armazenamento de arquivos |
| `CREATE MODEL` | Registrar `price_forecaster` e `purchase_optimizer` no registry |

!!! warning "Atenção: o catálogo deve existir antes de rodar o setup"
    O `00_setup.py` faz uma verificação explícita:

    ```python
    catalogs = [r[0] for r in spark.sql("SHOW CATALOGS").collect()]
    assert CATALOG in catalogs, (
        f"Catalog '{CATALOG}' not found. Set the 'catalog' widget to an existing Unity "
        f"Catalog you can write to (available: {catalogs})."
    )
    ```

    Se o catálogo configurado não existir, o notebook falha com uma mensagem clara (não com um erro críptico do registry). O catálogo padrão é `main`. Se você não tiver acesso ao `main`, altere `CATALOG` em `_config.py` para um catálogo em que tenha as permissões listadas acima antes de rodar qualquer notebook.

!!! warning "Atenção: permissões para modelos MLflow no Unity Catalog"
    `CREATE MODEL` é uma permissão separada de `CREATE TABLE`. Em alguns workspaces configurados de forma restritiva, um usuário pode ter permissão para criar tabelas mas não modelos. Se o `02_train_forecaster_sklearn.py` falhar ao registrar o modelo, verifique com o administrador se `CREATE MODEL` está concedida no schema.

---

## 3. Computação: serverless vs. cluster clássico

O workshop funciona nas duas modalidades, mas recomenda **serverless** com força.

### Por que serverless é recomendado

Cada notebook carrega um bloco de metadados **PEP 723** logo após o cabeçalho `# Databricks notebook source`:

```python
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///
```

Esse bloco instrui o ambiente serverless a usar o **ML base environment** (`databricks_ml_v5`), que já inclui MLflow, scikit-learn, pandas, numpy e outras dependências de ML, sem instalação manual. Cada notebook declara apenas seus **extras** além do base: os notebooks `03_register_optimizer_pyomo.py`, `04_end_to_end.py` e `05_verify.py` adicionam `pyomo` e `highspy`; o notebook opcional `extra/train_forecaster_pytorch.py` adiciona `torch` (CPU-only, para evitar o volume do wheel CUDA).

!!! note "Conceito: PEP 723, dependências inline no código-fonte"
    PEP 723 é uma proposta da comunidade Python que padroniza a declaração de dependências diretamente em scripts Python, sem `requirements.txt` separado e sem ambiente virtual manual. Na Databricks, o serverless interpreta esse bloco e provisiona o ambiente certo antes de executar o notebook. O resultado prático: **você nunca vê um `%pip install` nos notebooks principais**, pois o ambiente já está pronto quando a primeira célula roda.

!!! tip "Curiosidade: por que o base environment importa"
    O `databricks_ml_v5` é um snapshot versionado e testado do ecossistema de ML. Usar `environment_version = "5"` garante que o workshop rode com exatamente as mesmas versões de biblioteca, independentemente de quando você executar: hoje ou daqui a seis meses. Isso é **reprodutibilidade por design**.

### Cluster clássico (fallback)

Em clusters clássicos (não-serverless), o bloco PEP 723 é simplesmente ignorado. Nesse caso, instale as dependências manualmente com `%pip` no topo de cada notebook, seguido de `%restart_python` para aplicar o novo ambiente:

```python
%pip install -q -r requirements.txt
%restart_python
```

Use `requirements.txt` (na raiz do repositório) para os notebooks mandatórios (`00` a `05`), e `../requirements.txt` para os notebooks em `extra/`.

!!! warning "Atenção: não misture as duas abordagens"
    Em serverless, **não adicione** células `%pip install` manualmente: o PEP 723 já gerencia o ambiente. Adicionar um `%pip install` no meio de um notebook serverless reinicia o processo Python e pode causar comportamento inesperado.

---

## 4. Importe os notebooks como Git folder

Para que os comandos `%run ./_config` e `import workshop_lib` funcionem corretamente, os notebooks precisam estar em uma **Git folder** no workspace, não enviados manualmente um por um.

**Como fazer:**

1. Na barra lateral do workspace, clique em **Workspace** e navegue até onde quer criar a pasta.
2. Clique em **Adicionar** → **Git folder**.
3. Cole a URL do repositório e selecione o branch **`main`**.
4. Confirme a criação da pasta.

Isso clona o repositório inteiro como uma unidade. O `%run ./_config` (que todos os notebooks executam logo no início) usa um caminho relativo (`./_config`), e o `import workshop_lib` busca o módulo no diretório atual. Ambos resolvem corretamente quando os arquivos estão lado a lado na mesma Git folder, como no repositório.

!!! warning "Atenção: não faça upload avulso de notebooks"
    Se você fizer upload dos `.py` individualmente (sem a estrutura de pasta do repositório), o `%run ./_config` vai falhar com `FileNotFoundError` e o `import workshop_lib` vai falhar com `ModuleNotFoundError`. A Git folder é a única abordagem que garante que os caminhos relativos e os imports funcionem.

---

## 5. Configure uma única vez

Abra `_config.py` na Git folder e ajuste as duas constantes no topo:

```python
# 👉 Set these to a Unity Catalog + schema you can write to. Edit once; every
# notebook picks it up via `%run ./_config`.
CATALOG = "main"
SCHEMA  = "mlops_workshop"
```

O `_config.py` deriva todos os outros nomes a partir dessas duas constantes:

```python
FORECASTER_MODEL = f"{CATALOG}.{SCHEMA}.price_forecaster"
OPTIMIZER_MODEL  = f"{CATALOG}.{SCHEMA}.purchase_optimizer"
DATA_TABLE       = f"{CATALOG}.{SCHEMA}.commodity_monthly"
CSV_PATH         = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/commodity_monthly.csv"
```

Todos os notebooks fazem `%run ./_config` como primeira célula de código, então **editar essas duas linhas uma única vez propaga a mudança por todo o workshop**. Nenhum notebook tem URLs de workspace, tokens, ou caminhos `/dbfs/` hardcoded: apenas referências a essas constantes.

!!! tip "Curiosidade: por que `/Volumes/` e não `/dbfs/`"
    Em compute serverless, o DBFS FUSE (que permite acessar arquivos em `/dbfs/`) não está disponível. O workshop usa UC Volumes, uma abstração de armazenamento de arquivos governada pelo Unity Catalog, acessível via `/Volumes/{catalog}/{schema}/{volume}/`. Volumes funcionam tanto em serverless quanto em clusters clássicos, tornando os notebooks portáveis.

---

## 6. Execute o setup

Com tudo acima pronto, execute o notebook **`00_setup.py`**. Ele:

1. Faz `%run ./_config` para carregar as constantes.
2. Configura `mlflow.set_registry_uri("databricks-uc")` para apontar o registry ao Unity Catalog.
3. Verifica que o catálogo `CATALOG` existe (falha com mensagem clara se não).
4. Cria o schema `SCHEMA` e o Volume `workshop_files` (idempotente, seguro re-executar).
5. Registra o experimento MLflow em `/Shared/mlops_workshop`.

!!! success "Pronto quando…"
    O `00_setup.py` imprimir a linha final:

    ```
    Setup complete. Using main.mlops_workshop.
    ```

    (ou o catálogo/schema que você configurou). Quando isso aparecer, o ambiente está pronto e você pode seguir para o Lab 1.

---

## Resumo dos pré-requisitos

| Item | Detalhe |
|---|---|
| Workspace com Unity Catalog | Catálogo acessível com permissões de escrita |
| Permissões no catálogo | `USE CATALOG`, `CREATE SCHEMA`, `CREATE TABLE`, `CREATE VOLUME`, `CREATE MODEL` |
| Computação | Serverless (recomendado) ou cluster com Databricks Runtime ML |
| Notebooks | Importados como Git folder do branch `main` |
| Configuração | `CATALOG` e `SCHEMA` editados em `_config.py` |

---

**Próximo passo:** [Gerar o dataset sintético](../lab-1-generate-data/index.md)
