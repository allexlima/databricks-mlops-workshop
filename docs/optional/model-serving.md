# Model Serving: publicando o champion em tempo real

Este lab expõe o `price_forecaster@champion` como um endpoint HTTP gerenciado na Databricks, capaz de receber features de entrada e devolver previsões de preço em tempo real.

Abra `extra/serving.py`. Documentação de referência: [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html).

---

## O que é Model Serving na Databricks

O Databricks Model Serving isola cada modelo em um container gerenciado: instala automaticamente as dependências registradas com o modelo (`pip_requirements`), carrega os pesos, e expõe tudo via uma API REST autenticada. Você não gerencia infraestrutura, Dockerfile nem escalamento manual: define o tamanho da workload e o Databricks cuida do resto.

!!! note "Conceito"
    Quando você registrou o `price_forecaster` em `02_train_forecaster_sklearn.py`, o MLflow salvou junto os `pip_requirements` necessários (sklearn, numpy, etc.). O container de serving usa exatamente essa lista para montar o ambiente de inferência, garantindo que o modelo em produção rode com as mesmas versões de bibliotecas com que foi treinado. Isso é reprodutibilidade de ponta a ponta.

---

## Resolvendo a versão exata pelo alias

O notebook não hardcoda um número de versão. Ele resolve qual versão está atualmente marcada como `champion` no Unity Catalog e usa esse número para criar o endpoint:

```python
from mlflow.deployments import get_deploy_client
from mlflow import MlflowClient

client = MlflowClient()
deploy = get_deploy_client("databricks")

champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
```

Isso garante que o endpoint sempre aponte para a versão promovida mais recente. Se você rodar o lab PyTorch depois e promover uma nova versão para `@champion`, bastaria recriar (ou atualizar) o endpoint para ele refletir a mudança, sem alterar nenhuma linha de código.

!!! warning "Atenção"
    O notebook pressupõe que `FORECASTER_MODEL` tem um alias `@champion` definido. Se o `02_train_forecaster_sklearn.py` não tiver sido concluído com sucesso, `get_model_version_by_alias` lançará um erro. Confirme que o forecaster passou o validation gate antes de rodar este notebook.

---

## Criando o endpoint com scale-to-zero

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

!!! info "📸 Screenshot"
    *Reservado: capture aqui o endpoint `mlops-workshop-forecaster` com status Ready na aba Serving do workspace.*

---

## Consultando o endpoint

Quando o status for `Ready`, você pode enviar requisições diretamente pela aba Serving (botão "Query endpoint") ou via REST com autenticação por token. O payload de entrada deve conter as colunas de drivers e `price` no formato JSON que o MLflow DatasetInput espera.

!!! tip "Curiosidade"
    O endpoint criado pelo `mlflow.deployments.get_deploy_client("databricks")` é automaticamente protegido pelo mecanismo de autenticação do workspace. Não é preciso configurar API keys separadas: o token do workspace (ou um service principal) já é suficiente para autenticar as requisições.

---

## O optimizer Pyomo também pode ser servido

O mesmo padrão funciona para o `purchase_optimizer`: basta criar um segundo endpoint apontando para `OPTIMIZER_MODEL`. As dependências `pyomo` e `highspy` já foram declaradas nos `pip_requirements` quando o modelo foi registrado em `03_register_optimizer_pyomo.py`; o container de serving as instala automaticamente.

!!! info "Servindo o optimizer"
    Para servir o optimizer, crie um segundo endpoint com `entity_name=OPTIMIZER_MODEL` e o mesmo padrão de configuração. O `highspy` (solver HiGHS) é puramente pip-instalável e já está declarado nos `pip_requirements` do modelo registrado em `03_register_optimizer_pyomo.py`. Ver mais em [Registrar e promover a @champion](../lab-3-optimizer/registrar-promover.md).

!!! success "Pronto quando..."
    - A célula de criação imprime `Creating endpoint mlops-workshop-forecaster for <model> v<N>.`
    - O status do endpoint chega em **Ready** na aba Serving dentro de alguns minutos.
    - Uma query REST com colunas de drivers e `price` como JSON retorna um valor numérico em `predictions`.

---

**Próximo passo (opcional):** [Limpeza dos recursos](cleanup.md)
