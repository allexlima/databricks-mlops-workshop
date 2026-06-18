# Labs opcionais

O caminho obrigatório do workshop vai do Lab 1 ao Lab 5: gerar dados → treinar o forecaster sklearn → registrar o optimizer Pyomo → compor a cadeia → verificar. Os três notebooks abaixo são **inteiramente opcionais e independentes entre si**: você pode rodar qualquer um deles em qualquer ordem, sem afetar os labs principais.

| Notebook | O que faz | Pré-requisito |
|---|---|---|
| `extra/train_forecaster_pytorch.py` | Mesmo lifecycle, framework diferente | Lab 2 concluído (tabela de dados existe) |
| `extra/serving.py` | Publica o forecaster em um endpoint em tempo real | `@champion` definido no `price_forecaster` |
| `extra/cleanup.py` | Remove todos os recursos do workshop | Nenhum (mas leia o aviso antes de rodar) |

---

## PyTorch Forecaster: mesmo lifecycle, outro framework

### Por que este lab existe

O Lab 2 treina um gradient boosting com scikit-learn. Este interlude mostra que o **MLflow governa o que você tiver**, não apenas sklearn. Basta trocar o objeto do modelo; o restante do ciclo (experiment tracking, validation gate, registro no Unity Catalog, alias `@champion`) permanece **byte a byte idêntico**. Isso é exatamente a tese central do workshop aplicada a um segundo framework.

Abra `extra/train_forecaster_pytorch.py`.

!!! note "Conceito"
    Um framework de deep learning como PyTorch expõe seus modelos como objetos Python comuns (`nn.Module`). O MLflow tem um flavor nativo, `mlflow.pytorch`, que sabe serializar e desserializar esses objetos automaticamente, incluindo os pesos treinados. Na hora de carregar o modelo, você recebe de volta um `nn.Module` funcional, sem precisar reescrever nenhuma classe.

### O modelo: MLP de duas camadas

O notebook define um pequeno perceptron multicamada (MLP) totalmente conectado: duas camadas lineares separadas por uma ativação ReLU, com 32 neurônios na camada oculta. É deliberadamente simples: o foco é o lifecycle, não a arquitetura.

```python
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)
```

O treinamento usa Adam por 400 épocas com `lr=0.01` e MSE como função de perda. A semente aleatória é fixada via `torch.manual_seed(SEED)`, o mesmo `SEED` do `_config.py` que todos os notebooks herdam, para garantir reprodutibilidade.

### Dados, split e normalização: idênticos ao Lab 2

O notebook lê a mesma tabela Delta criada no Lab 1 (`DATA_TABLE`), aplica o mesmo corte cronológico 80/20, e normaliza as features com z-score calculado **exclusivamente sobre os dados de treino** (média e desvio padrão do conjunto de treino). Usar estatísticas do conjunto de teste na normalização seria data leakage, um erro clássico em pipelines de ML.

```python
mu, sd = train[feats].mean(), train[feats].std(ddof=0)

def to_t(frame):
    return torch.tensor(((frame[feats] - mu) / sd).to_numpy(), dtype=torch.float32)
```

!!! tip "Curiosidade"
    O z-score é simples, mas eficaz para redes neurais: gradientes explodem ou somem quando as entradas têm escalas muito diferentes. Normalizar antes do treino é um dos ajustes de maior impacto para estabilizar a convergência de MLPs, mais do que a maioria dos hiperparâmetros de arquitetura.

### Tracking e log: dentro de um único `mlflow.start_run`

Todo o treinamento acontece dentro de um único bloco `with mlflow.start_run(run_name="pytorch_mlp")`. Isso garante que parâmetros, métricas e o artefato do modelo pertençam à mesma run, e que o `info.model_uri` devolvido por `log_model` seja a URI correta para o passo de registro.

```python
with mlflow.start_run(run_name="pytorch_mlp"):
    # ... loop de treino ...
    mlflow.log_params({"model_type": "torch_mlp", "epochs": 400, "seed": SEED})
    mlflow.log_metrics({"r2": r2, "rmse": rmse})
    info = mlflow.pytorch.log_model(net, name="model")
```

O experimento é o mesmo que o Lab 2 usa, definido em `EXPERIMENT_PATH` no `_config.py`. Isso significa que as runs do sklearn e do PyTorch aparecem **lado a lado** na mesma view do MLflow, facilitando a comparação de métricas.

### Validation gate e promoção para `@champion`

O mesmo `R2_THRESHOLD` (0.6) do Lab 2. Se o modelo passar, ele é registrado no Unity Catalog com o mesmo nome `FORECASTER_MODEL` e recebe o alias `@champion`, **podendo sobrescrever a versão sklearn anterior**.

```python
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(info.model_uri, FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate. Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f}). Not promoted.")
```

Isso é intencional: diferentes frameworks competindo pelo mesmo alias de produção. O Lab 4 carrega o `@champion` pelo alias (não pelo número de versão), então ele funcionará independentemente de qual framework está apontado.

!!! note "Conceito"
    Este é o poder dos aliases no Unity Catalog: desacoplam o código consumidor da versão concreta. O Lab 4 sempre chama `client.get_model_version_by_alias(FORECASTER_MODEL, "champion")`, nunca `v3` ou `v7`. Quem promoveu qual versão é detalhe de governança, não de código.

### Dependência de compute: `torch` no PEP 723

O notebook declara `torch` como dependência extra no bloco PEP 723, que é a única diferença de ambiente em relação ao Lab 2. O torch é pinado para download CPU-only para evitar o peso das wheels CUDA (vários gigabytes), que seriam desnecessários numa demo em serverless.

```
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# dependencies = [
#   "torch",
# ]
# ///
```

!!! warning "Atenção"
    Em clusters clássicos, o bloco PEP 723 é ignorado. Use `%pip install -q torch --index-url https://download.pytorch.org/whl/cpu` seguido de `%restart_python` no início do notebook antes de rodar.

!!! success "Pronto quando…"
    - A saída mostra `PASSED gate. Registered v<N> as @champion.` (ou `FAILED gate (r2=...)`, aceitável se o ambiente for menos estável; tente aumentar épocas).
    - Uma run chamada `pytorch_mlp` aparece no experimento MLflow compartilhado.
    - O Unity Catalog mostra `FORECASTER_MODEL@champion` apontando para a nova versão PyTorch.

---

## Model Serving: publicando o champion em tempo real

### O que o lab faz

Este notebook pega o `price_forecaster@champion` (seja o sklearn ou o PyTorch, dependendo do que foi promovido por último) e o publica como um **endpoint de serving em tempo real** na Databricks. O endpoint recebe requisições HTTP com os drivers de commodity (features) e devolve a previsão de preço em JSON.

Abra `extra/serving.py`.

!!! note "Conceito"
    O Databricks Model Serving isola cada modelo em um container gerenciado, instala as dependências que foram registradas com o modelo (`pip_requirements`), carrega os pesos, e expõe tudo via uma API REST autenticada. Você não precisa gerenciar infraestrutura, Dockerfile, nem escalamento manual.

### Resolvendo a versão exata pelo alias

O notebook não hardcoda um número de versão. Ele resolve qual versão está atualmente marcada como `champion` no Unity Catalog e usa esse número para criar o endpoint:

```python
from mlflow.deployments import get_deploy_client
from mlflow import MlflowClient

client = MlflowClient()
deploy = get_deploy_client("databricks")

champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
```

Isso garante que o endpoint sempre aponte para a versão promovida mais recente, não para uma versão arbitrária hardcodada no código.

### Criando o endpoint com scale-to-zero

```python
endpoint = "mlops-workshop-forecaster"
deploy.create_endpoint(
    name=endpoint,
    config={
        "served_entities": [
            {
                "entity_name": FORECASTER_MODEL,
                "entity_version": champ.version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ]
    },
)
print(f"Creating endpoint {endpoint} for {FORECASTER_MODEL} v{champ.version}.")
```

`scale_to_zero_enabled: True` faz o endpoint desligar automaticamente quando fica ocioso, sem custo de compute enquanto não há requisições. Ideal para demos que não estão em produção contínua.

!!! tip "Curiosidade"
    O provisionamento de um novo endpoint leva alguns minutos após a chamada `create_endpoint` retornar: o Databricks está inicializando o container, instalando dependências e carregando o modelo. Acompanhe o progresso na aba **Serving** do workspace. O status transita de `Not Ready` para `Ready` quando o endpoint está apto a receber tráfego.

### O optimizer Pyomo também pode ser servido

O mesmo padrão funciona para o `purchase_optimizer`: basta criar um segundo endpoint apontando para `OPTIMIZER_MODEL`. As dependências `pyomo` e `highspy` já foram declaradas nos `pip_requirements` quando o modelo foi registrado no Lab 3; o container de serving as instala automaticamente.

!!! info "Servindo o optimizer"
    Para servir o optimizer, crie um segundo endpoint com `entity_name=OPTIMIZER_MODEL` e o mesmo padrão de configuração. O `highspy` (solver HiGHS) é puramente pip-instalável e já está declarado nos `pip_requirements` do modelo registrado no Lab 3.

!!! info "Screenshot"
    *Capture aqui: o endpoint `mlops-workshop-forecaster` com status Ready na aba Serving. Depois substitua por `![Endpoint pronto](../assets/screenshots/opcional-serving.png)`.*

!!! success "Pronto quando…"
    - A célula de criação imprime `Creating endpoint mlops-workshop-forecaster for <model> v<N>.`
    - O status do endpoint chega em **Ready** na aba Serving dentro de alguns minutos.
    - Uma query REST com colunas de drivers e `price` como JSON retorna um valor numérico em `predictions`.

---

## Cleanup: removendo todos os recursos do workshop

### O que o notebook remove

`extra/cleanup.py` é **destrutivo por design**: apaga o endpoint de serving, os dois modelos registrados no Unity Catalog (`FORECASTER_MODEL` e `OPTIMIZER_MODEL`) com todas as suas versões, o experimento MLflow com todas as runs, a tabela Delta, o Volume CSV e, por padrão, o schema inteiro via `DROP SCHEMA ... CASCADE`.

Cada passo de deleção é independente e protegido por `try/except`: se o lab de serving foi pulado e o endpoint não existe, o notebook continua normalmente para os demais recursos.

Abra `extra/cleanup.py`.

!!! warning "Atenção"
    Este notebook **deleta permanentemente** todos os recursos criados pelo workshop. A ação não pode ser desfeita. Leia os dois flags de controle antes de executar qualquer célula.

### Dois flags de controle: leia antes de rodar

```python
CONFIRM = True       # 👉 defina False para abortar antes de qualquer deleção
DROP_SCHEMA = True   # 👉 defina False para manter o schema
```

| Flag | Padrão | Efeito |
|---|---|---|
| `CONFIRM` | `True` | `False` aborta imediatamente via `assert` (nada é tocado). |
| `DROP_SCHEMA` | `True` | `False` preserva o schema; remove apenas modelos, tabela e volume. |

O `assert CONFIRM` é a primeira instrução executável do notebook. Se você mudar para `False`, toda a execução para ali, seguramente.

### O que cada passo apaga

```python
# 1. Endpoint de serving
get_deploy_client("databricks").delete_endpoint("mlops-workshop-forecaster")

# 2. Modelos registrados (com todas as versões)
for name in (FORECASTER_MODEL, OPTIMIZER_MODEL):
    client.delete_registered_model(name)

# 3. Experimento MLflow e todas as runs
mlflow.delete_experiment(exp.experiment_id)

# 4. Tabela Delta e Volume CSV
spark.sql(f"DROP TABLE IF EXISTS {DATA_TABLE}")
spark.sql(f"DROP VOLUME IF EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")

# 5. Schema (se DROP_SCHEMA = True)
spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
```

!!! warning "Atenção ao schema compartilhado"
    Se o catalog ou schema for usado por outros participantes, outros workshops, ou qualquer outra carga de trabalho, **defina `DROP_SCHEMA = False`** antes de executar. O `DROP SCHEMA ... CASCADE` remove tudo dentro do schema sem confirmação adicional do Databricks, inclusive objetos que não foram criados por este workshop.

!!! success "Pronto quando…"
    - `Deleted serving endpoint.` (ou `No serving endpoint to delete (…).` se o lab de serving foi pulado).
    - `Deleted registered model <name>.` para `FORECASTER_MODEL` e `OPTIMIZER_MODEL`.
    - `Deleted experiment <path>.`
    - `Dropped <DATA_TABLE> and volume <VOLUME>.`
    - `Dropped schema <CATALOG>.<SCHEMA>.` (apenas se `DROP_SCHEMA = True`).
    - Última linha: `Cleanup complete.`

---

Pronto com os labs opcionais? Volte para [Verificação](../lab-5-verify/index.md) se quiser revisar os checkpoints do caminho principal.
