# Labs opcionais

Estes três labs são **opcionais e não fazem parte do caminho obrigatório do workshop** — cada um é totalmente independente dos demais e pode ser executado em qualquer ordem, sem afetar os Labs 1–4.

---

## Forecaster em PyTorch

### O que é e por quê fazer

O objetivo deste lab é mostrar que o ciclo de vida gerenciado pelo MLflow é **agnóstico ao framework**: basta trocar o objeto do modelo — o restante do fluxo permanece idêntico ao lab do sklearn. Aqui substituímos o gradient boosting por um pequeno MLP totalmente conectado (duas camadas lineares com ReLU), treinado com Adam por 400 épocas.

Os dados, o split cronológico 80/20, a normalização z-score (calculada apenas com estatísticas do treino) e o portão de validação com `R2_THRESHOLD` são **exatamente os mesmos** do lab principal. Se o modelo passar no portão, ele é registrado no Unity Catalog e recebe o alias `@champion` — podendo sobrescrever o modelo sklearn anterior. Isso é intencional: demonstra que diferentes frameworks competem pelo mesmo alias de produção.

!!! info "Não é dependência da cadeia ponta a ponta"
    Este notebook **não precisa rodar antes ou depois** do lab sklearn. Ele reutiliza o mesmo experimento MLflow, o mesmo portão de validação e pode reapontar o `@champion` livremente.

Abra `extra/train_forecaster_pytorch.py`.

```python
# MLP de duas camadas treinado com Adam por 400 épocas
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

with mlflow.start_run(run_name="pytorch_mlp"):
    # ... treinamento ...
    mlflow.log_metrics({"r2": r2, "rmse": rmse})
    info = mlflow.pytorch.log_model(net, name="model")

# Mesmo portão do lab sklearn
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(info.model_uri, FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
```

!!! success "Checkpoint — o que você deve ver"
    - `PASSED gate. Registered v<N> as @champion.` — ou `FAILED gate (r2=...)` caso o R² fique abaixo do limiar (pode acontecer dependendo do ambiente; tente aumentar épocas ou ajustar o learning rate).
    - Uma nova run chamada `pytorch_mlp` visível no experimento MLflow compartilhado.
    - `FORECASTER_MODEL@champion` apontando para a versão PyTorch no Unity Catalog.

---

## Model Serving

### O que é e por quê fazer

Este lab publica o modelo `price_forecaster@champion` em um **endpoint de serving em tempo real** no Databricks Model Serving. O endpoint é configurado com CPU e **scale-to-zero** habilitado, o que significa que ele não gera custo quando está ocioso entre execuções de demo. O provisionamento leva alguns minutos após a chamada `create_endpoint` retornar — acompanhe o status na aba **Serving** do workspace.

Além do forecaster, o otimizador Pyomo também pode ser servido da mesma forma: seus `pip_requirements` já incluem `highspy` e `pyomo`, ambos instaláveis via pip dentro do container de serving.

!!! info "Pyomo também pode ser servido"
    Para servir o otimizador, basta criar um segundo endpoint apontando para `OPTIMIZER_MODEL` com o mesmo padrão — o highspy é instalável via pip no container e já está declarado nos `pip_requirements` do modelo registrado.

Abra `extra/serving.py`.

```python
# Resolve a versão exata marcada como @champion no Unity Catalog
champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")

deploy.create_endpoint(
    name="mlops-workshop-forecaster",
    config={
        "served_entities": [{
            "entity_name": FORECASTER_MODEL,
            "entity_version": champ.version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True,
        }]
    },
)
```

!!! note "📸 Espaço reservado para captura de tela"
    *Capture aqui: o endpoint de serving com status Ready. Depois substitua por `![Endpoint pronto](../assets/screenshots/opcional-serving.png)`.*

!!! success "Checkpoint — o que você deve ver"
    - `Creating endpoint mlops-workshop-forecaster for <model> v<N>.` impresso imediatamente após a chamada.
    - Status do endpoint transitando para **Ready** na aba Serving dentro de alguns minutos.
    - Uma query REST bem-sucedida (colunas de drivers + price como JSON) retornando um valor predito de `price_next_month`.

---

## Limpeza / teardown

### O que é e por quê fazer

Este notebook remove permanentemente **todos os recursos criados pelo workshop**: o endpoint de serving, os dois modelos registrados (`FORECASTER_MODEL` e `OPTIMIZER_MODEL`), o experimento MLflow com todas as suas runs, a tabela Delta, o Volume CSV e, por padrão, o schema inteiro via `CASCADE`.

Cada etapa de deleção está envolvida em um `try/except` independente — se um recurso não existir (por exemplo, o endpoint de serving nunca foi criado porque o lab anterior foi pulado), a limpeza continua normalmente para os demais recursos.

!!! danger "DESTRUTIVO — esta ação não pode ser desfeita"
    Por padrão, **tudo é deletado**, incluindo o schema. Dois flags no topo do notebook controlam o comportamento:

    | Flag | Valor padrão | Efeito |
    |------|-------------|--------|
    | `CONFIRM` | `True` | Defina `False` para abortar antes de qualquer deleção. O `assert` dispara imediatamente e nada é tocado. |
    | `DROP_SCHEMA` | `True` | Defina `False` para preservar o schema e remover apenas modelos, tabela e volume. Use esta opção se o catalog ou schema for compartilhado com outros participantes. |

Abra `extra/cleanup.py`.

```python
CONFIRM = True       # 👉 defina False para abortar antes de qualquer deleção
DROP_SCHEMA = True   # 👉 defina False para manter o schema

assert CONFIRM, "CONFIRM is False — aborting teardown. Set CONFIRM = True to delete."

# ...
# Ao final, o schema é removido com CASCADE (se DROP_SCHEMA = True)
if DROP_SCHEMA:
    spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{SCHEMA} CASCADE")
```

!!! warning "Atenção ao schema compartilhado"
    Se o catalog ou schema for usado por outros participantes ou workshops, **defina `DROP_SCHEMA = False`** antes de executar. O `DROP SCHEMA ... CASCADE` remove tudo dentro do schema sem confirmação adicional.

!!! success "Checkpoint — o que você deve ver"
    - `Deleted serving endpoint.` — ou `No serving endpoint to delete (…).` se o lab de serving foi pulado.
    - `Deleted registered model <name>.` para cada um dos dois modelos.
    - `Deleted experiment <path>.`
    - `Dropped <DATA_TABLE> and volume <VOLUME>.`
    - `Dropped schema <CATALOG>.<SCHEMA>.` (apenas se `DROP_SCHEMA = True`).
    - Linha final: `Cleanup complete.`
