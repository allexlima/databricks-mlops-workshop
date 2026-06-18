# Clonar o repositório e configurar o ambiente

Os notebooks do workshop vivem no GitHub. Para rodá-los na Databricks, você vai
trazer o repositório para dentro do seu workspace como uma **Git folder**: o
recurso também conhecido como Repos. Esse passo garante que a estrutura de
pastas do repositório seja preservada, o que é essencial para que o
`%run ./_config` e o `import workshop_lib` resolvam os caminhos corretamente.

!!! note "Conceito"
    Uma **Git folder** é um clone de um repositório Git que vive dentro do
    workspace da Databricks. Ela mantém a hierarquia de diretórios idêntica à
    do repositório remoto. Isso importa porque o `%run ./_config` usa um caminho
    relativo (`./`), se você importar os notebooks soltos, sem essa estrutura,
    o Databricks não consegue encontrar o arquivo e o comando falha. Documentação
    oficial: [Git folders (Repos)](https://docs.databricks.com/en/repos/index.html).

---

## 1. Clonar como Git folder

Siga os passos abaixo no seu workspace:

1. Copie a URL do repositório:

    ```text
    https://github.com/allexlima/databricks-mlops-workshop.git
    ```

2. Na barra lateral, clique em **Workspace**. Navegue até a pasta onde quer
   guardar o material, por exemplo, a sua pasta de usuário (`/Users/<seu-email>/`).

3. Clique em **Create** (ou no menu de contexto com o botão direito) e escolha
   **Git folder**.

4. Cole a URL no campo **Git repository URL**. O provedor (**GitHub**) e o nome
   sugerido para a pasta (`databricks-mlops-workshop`) são preenchidos
   automaticamente.

5. Mantenha a branch **`main`** selecionada, é onde ficam todos os notebooks do
   caminho obrigatório.

6. Clique em **Create Git folder**. A Databricks clona o repositório e abre a
   pasta. Você vai ver os notebooks `00_setup` até `05_verify` na raiz e a pasta
   `extra/` com os labs opcionais.

!!! info "📸 Screenshot"
    *Reservado:* o diálogo "Create Git folder" com a URL
    `https://github.com/allexlima/databricks-mlops-workshop.git` preenchida e o
    campo de branch mostrando `main`.

!!! tip "Curiosidade"
    Como o repositório é público, você não precisa configurar credenciais de Git
    para cloná-lo. Credenciais (um personal access token em **Settings › Linked
    accounts**) só são necessárias se você quiser dar `pull` em atualizações
    futuras ou trabalhar com um fork privado seu.

### Como o repositório está organizado

| Caminho | O que é |
|---------|---------|
| `_config.py` | Constantes editáveis (`CATALOG`, `SCHEMA`), nomes derivados, `SEED`, `R2_THRESHOLD` |
| `workshop_lib.py` | Funções compartilhadas (`generate_dataset`, `solve_purchase`, `PurchaseOptimizerModel`) |
| `00_setup.py` … `05_verify.py` | Caminho obrigatório, na ordem |
| `extra/` | Labs opcionais: PyTorch, serving e cleanup |
| `requirements.txt` | Dependências para clusters clássicos |

---

## 2. Escolher o compute

O workshop roda em dois tipos de compute. Escolha o seu cenário:

=== "Serverless (recomendado)"

    O compute serverless da Databricks isola as dependências de cada notebook
    via metadados PEP 723 embutidos no próprio arquivo `.py`. Cada notebook já
    declara o ML base environment (`databricks_ml_v5`, versão `5`), que inclui
    MLflow, scikit-learn, pandas e numpy pré-instalados. Extras específicos
    (Pyomo e HiGHS nos notebooks de otimização) são declarados ali e instalados
    automaticamente quando o notebook roda.

    **Você não precisa instalar nada.** Basta selecionar um serverless compute no
    seletor de cluster no topo do notebook e executar as células.

    Documentação oficial:
    [Serverless compute](https://docs.databricks.com/en/compute/serverless/index.html).

    !!! note "Conceito"
        **PEP 723** é um padrão Python que permite declarar dependências de um
        script diretamente no cabeçalho do arquivo, em um bloco de metadados. A
        Databricks lê esse bloco antes de executar cada notebook serverless e
        monta um ambiente isolado com as libs declaradas. Notebooks clássicos
        ignoram esse bloco, por isso o procedimento de instalação é diferente.

=== "Cluster clássico"

    Em clusters clássicos (ML Runtime), o PEP 723 é ignorado. Você precisa
    instalar as dependências manualmente. Adicione as duas linhas abaixo como
    **primeira célula** de cada notebook antes de executar qualquer outro código:

    ```python
    %pip install -q -r requirements.txt   # a partir dos notebooks da raiz
    %restart_python
    ```

    Se estiver rodando os notebooks opcionais da pasta `extra/`, ajuste o
    caminho relativo:

    ```python
    %pip install -q -r ../requirements.txt
    %restart_python
    ```

    O `%restart_python` é obrigatório: ele reinicia o interpretador Python para
    que as libs recém-instaladas fiquem disponíveis na sessão atual.

    !!! warning "Atenção"
        Nunca pule o `%restart_python` após o `%pip install`. Sem ele, o Python
        continua usando a sessão anterior e as libs novas não são reconhecidas ,
        o import vai falhar ou silenciosamente usar uma versão antiga.

---

## 3. Configurar o `_config.py`

O arquivo `_config.py` é o único lugar onde você configura o workshop. Edite
as duas constantes no topo **uma única vez**: todos os notebooks fazem
`%run ./_config` e herdam esses valores automaticamente.

Abra o `_config.py` na raiz da Git folder e localize o bloco abaixo:

```python
# 👉 Set these to a Unity Catalog + schema you can write to. Edit once; every
# notebook picks it up via `%run ./_config`.
CATALOG = "main"
SCHEMA  = "mlops_workshop"
```

- **`CATALOG`**: nome do catálogo Unity Catalog onde o workshop vai criar seus
  objetos. O valor padrão é `main`. Troque pelo catálogo ao qual você tem
  permissão de criar schemas. O catálogo precisa existir previamente, o
  `00_setup` vai checar isso e mostrar uma mensagem clara se não encontrar.
- **`SCHEMA`**: nome do schema que o workshop vai criar dentro do catálogo. O
  padrão `mlops_workshop` funciona bem para a maioria dos casos. Use um nome
  diferente se quiser isolar sua instância do workshop em um ambiente
  compartilhado.

A partir dessas duas constantes, o `_config.py` deriva todos os outros nomes
usados no workshop:

```python
EXPERIMENT_PATH  = "/Shared/mlops_workshop"
FORECASTER_MODEL = f"{CATALOG}.{SCHEMA}.price_forecaster"
OPTIMIZER_MODEL  = f"{CATALOG}.{SCHEMA}.purchase_optimizer"
DATA_TABLE       = f"{CATALOG}.{SCHEMA}.commodity_monthly"
VOLUME           = "workshop_files"
CSV_PATH         = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/commodity_monthly.csv"
```

!!! tip "Curiosidade"
    Todos os artefatos de arquivo (datasets CSV, modelos, etc.) são gravados em
    um **UC Volume** (`/Volumes/{catalog}/{schema}/workshop_files/…`), nunca em
    `/dbfs/`. O DBFS FUSE não está disponível no compute serverless, usar
    Volumes é a forma correta e portável de armazenar arquivos na Databricks
    moderna. Saiba mais sobre o
    [MLflow no Databricks](https://docs.databricks.com/en/mlflow/index.html).

!!! info "📸 Screenshot"
    *Reservado:* o arquivo `_config.py` aberto no editor do workspace, com as
    linhas `CATALOG` e `SCHEMA` destacadas no topo e os nomes derivados visíveis
    abaixo.

!!! warning "Atenção"
    Não inclua o nome do catálogo em variáveis hardcoded em nenhuma célula de
    notebook, sempre use as constantes do `_config`. Isso garante que o
    workshop funcione em qualquer workspace sem edições espalhadas.

---

## 4. Rodar o `00_setup.py`

Com o `_config.py` configurado, abra o notebook `00_setup.py` e execute todas
as células em ordem. O que ele faz, passo a passo:

1. **`%run ./_config`**: carrega todas as constantes do `_config.py` no
   escopo da sessão.
2. **`mlflow.set_registry_uri("databricks-uc")`**: aponta o MLflow para o
   Unity Catalog como registry, em vez do registry legado do workspace.
3. **Verifica que o catálogo existe**: faz um `SHOW CATALOGS` e interrompe
   com uma mensagem clara se `CATALOG` não for encontrado. O `00_setup` nunca
   tenta criar o catálogo, isso requer privilégios de admin e depende de
   configurações de storage que variam por workspace.
4. **Cria o schema e o volume**: `CREATE SCHEMA IF NOT EXISTS` e
   `CREATE VOLUME IF NOT EXISTS`, ambos idempotentes (seguro re-executar).
5. **Registra o experimento MLflow**: `mlflow.set_experiment(EXPERIMENT_PATH)`
   garante que todos os labs loguem runs no mesmo experimento, facilitando
   comparações na UI de Experiments.
6. **Imprime a confirmação final**: `Setup complete. Using <catalog>.<schema>.`

!!! info "📸 Screenshot"
    *Reservado:* a saída do `00_setup.py` no workspace mostrando a linha
    `Setup complete. Using main.mlops_workshop.` e, ao lado, o Catalog Explorer
    com o schema `mlops_workshop` e o volume `workshop_files` recém-criados.

!!! warning "Atenção"
    Se o `assert` do catálogo falhar, a mensagem vai listar os catálogos
    disponíveis para você. Basta atualizar o `CATALOG` no `_config.py` para um
    catálogo da lista e re-executar o `00_setup`.

!!! success "Pronto quando..."
    O `00_setup.py` imprime a linha:

    ```
    Setup complete. Using <catálogo>.<schema>.
    ```

    Isso confirma que o MLflow está apontado para o Unity Catalog, o schema e o
    volume existem, e o experimento foi registrado. Você está pronto para
    começar os labs.

---

Próximo passo: [Checklist do ambiente](prerequisites.md)
