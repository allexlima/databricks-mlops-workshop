# Pré-requisitos

Antes de começar os labs, garanta que o ambiente está pronto. Leva poucos minutos.

## O que você precisa

- Um workspace Databricks com **Unity Catalog** habilitado.
- Computação **serverless** (recomendada) ou um cluster clássico com **Databricks
  Runtime ML**.
- Um catálogo do Unity Catalog onde você consiga escrever e com permissão para
  **criar um schema**.

!!! warning "Permissões no Unity Catalog"
    Você precisa de `USE CATALOG` em um catálogo, mais `CREATE SCHEMA` (e, dentro dele,
    `CREATE TABLE` / `CREATE MODEL` / `CREATE VOLUME`). O notebook de setup **não cria
    um catálogo** — criar catálogos é uma operação privilegiada e depende das
    configurações de *managed location*. Ele só cria um schema e um volume **dentro** de
    um catálogo que já é seu. Se o catálogo padrão (`main`) não for seu, edite `CATALOG`
    no `_config` antes de rodar.

## Traga os notebooks

Importe os notebooks do workshop (o branch `main` do repositório) para o seu workspace
como uma **Git folder** — assim o `import workshop_lib` e o `%run ./_config` resolvem
corretamente.

## Configure uma única vez

Abra o `_config` e ajuste as duas constantes no topo para um catálogo/schema em que você
tenha permissão de escrita:

```python
CATALOG = "main"            # 👉 seu catálogo no Unity Catalog
SCHEMA  = "mlops_workshop"  # 👉 seu schema
```

Todos os notebooks fazem `%run` do `_config`, então você edita isso em um só lugar.

## Dependências

=== "Serverless (recomendado)"
    Nada a instalar. Cada notebook declara o seu ambiente como metadados **PEP 723** no
    topo do código-fonte (o *base environment* de ML, mais alguns extras quando
    necessário), e o ambiente *serverless* é provisionado automaticamente.

=== "Cluster clássico / ML"
    Em computação clássica o PEP 723 é ignorado. Rode isto uma vez no topo de cada
    notebook (ou instale as bibliotecas no cluster):

    ```python
    %pip install -q -r requirements.txt
    %restart_python
    ```

## Rode o setup

Execute o **`00_setup`** primeiro. Ele aponta o MLflow para o Unity Catalog, verifica se
o seu catálogo existe e cria o schema, o volume e o experimento.

!!! success "Tudo pronto quando…"
    O `00_setup` imprimir `Setup complete. Using <catálogo>.<schema>.` — então siga para
    o Lab 1.
