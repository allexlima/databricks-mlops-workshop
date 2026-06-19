# Treinar o forecaster (sklearn)

Com o dataset gerado no lab anterior, você tem 96 meses de preços de commodity e
dez indicadores econômicos sintéticos persistidos na tabela
`main.mlops_workshop_<seu-usuário>.commodity_monthly` do Unity Catalog. Este lab usa esses
dados como ponto de partida para percorrer o loop cotidiano de MLOps na
Databricks: treinar, rastrear, avaliar contra um critério objetivo e, somente se
o modelo passar, registrar e promover para `@champion`.

Promoção é um gesto deliberado. Não é efeito colateral do treinamento.

Abra `02_train_forecaster_sklearn.py`.

---

## O que acontece neste lab

| Passo | O que o código faz | Conceito MLOps |
|---|---|---|
| 1 | `mlflow.set_registry_uri("databricks-uc")` + `set_experiment(...)` | Apontar para o registry e o experiment corretos |
| 2 | Split cronológico 80/20 sem shuffle | Held-out set temporal, sem data leakage |
| 3 | `mlflow.start_run(...)` + fit + log | Tracking atômico: params, metrics, artefato |
| 4 | `r2 >= R2_THRESHOLD` → `register_model` → alias `@champion` | Validation gate determinístico |

Os Passos 1 a 3 são o bloco de tracking: configurar, preparar dados, executar o run.
O Passo 4 é o desfecho: avaliar o run e decidir se ele merece virar um modelo governado.

---

## Conexão com a etapa anterior

A [geração do conjunto de dados](../lab-1-generate-data/index.md) persistiu o dataset; esta etapa o consome.
A divisão cronológica 80/20 aplicada no Passo 2 segue o mesmo princípio de
*time-respecting split* visto no baseline ingênuo de `01_generate_data.py`: o held-out set deve
ser sempre as observações mais recentes, nunca uma amostra aleatória. Aqui o
princípio se aplica ao modelo real.

Se você ainda não rodou `01_generate_data.py`, faça isso antes de continuar.
O Passo 2 carrega `spark.table(DATA_TABLE)` e falha com erro de tabela não
encontrada se o dataset não existir.

---

## Páginas desta seção

| Página | O que cobre |
|---|---|
| [Tracking com MLflow](tracking.md) | Passos 1 a 3: configurar o MLflow, dividir os dados, treinar e logar o run |
| [Validation gate e registro](validation-gate.md) | Passo 4: gate de R², registrar no Unity Catalog, alias `@champion`, verificar no Catalog Explorer |

**Comece por:** [Tracking com MLflow](tracking.md)
