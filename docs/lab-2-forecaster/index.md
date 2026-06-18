# Lab 2 — Treinar o forecaster (sklearn)

Este lab é o coração do workshop: o loop cotidiano de MLOps na Databricks.
Você vai treinar um modelo de previsão de preço, rastrear tudo no MLflow,
e — só se o modelo passar por uma régua objetiva — registrá-lo no Unity
Catalog e promovê-lo ao alias `@champion`. Promoção é uma **decisão
deliberada**, não um efeito colateral do treinamento.

Abra `02_train_forecaster_sklearn.py`.

---

## Visão geral do que acontece neste notebook

| Passo | O que o código faz | Conceito MLOps |
|---|---|---|
| 1 | `mlflow.set_registry_uri("databricks-uc")` + `set_experiment(...)` | Apontar para o registry e o experiment certos |
| 2 | Split cronológico 80/20 sem shuffle | Held-out set temporal — sem data leakage |
| 3 | `mlflow.start_run(...)` + fit + log | Tracking atômico: params, metrics, artefato |
| 4 | `r2 >= R2_THRESHOLD` → `register_model` → alias `@champion` | Validation gate determinístico |

---

## Passo 1 · Apontar o MLflow para o Unity Catalog

```python
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)
```

!!! note "Conceito — dois endereços distintos no MLflow"
    O MLflow tem **duas áreas de armazenamento** que podem ser configuradas
    de forma independente:

    - **Tracking server** — onde os *runs* (experimentos) ficam: params,
      métricas, artefatos brutos. Na Databricks, o tracking server da
      plataforma é usado automaticamente; você não precisa apontar uma URL.
    - **Model Registry** — onde as versões de modelo promovidas ficam,
      com aliases, tags e lineage. É aqui que a linha `set_registry_uri`
      age: por padrão o registry é o *Workspace Model Registry* (legado).
      `"databricks-uc"` redireciona para o **Unity Catalog Registry**, que
      adiciona governança de dados (linhagem, ACLs por schema, auditoria
      completa) ao ciclo de vida do modelo.

    `EXPERIMENT_PATH` é `"/Shared/mlops_workshop"`, definido em `_config.py`.
    Usar um caminho compartilhado garante que todos os participantes do
    workshop vejam os runs no mesmo experiment — fica fácil comparar
    execuções lado a lado na UI do MLflow.

!!! tip "Curiosidade — por que Unity Catalog e não o registry antigo?"
    O Workspace Model Registry original não tem controle de acesso por
    modelo: quem pode ver o workspace pode ver todos os modelos. Com o
    Unity Catalog, você granulariza via `GRANT` no nível de schema ou de
    modelo individual. Além disso, a lineage automática conecta o modelo à
    tabela de origem (`commodity_monthly`) e ao run de treinamento — tudo
    visível no Catalog Explorer sem configuração extra.

---

## Passo 2 · Divisão treino/teste que respeita o tempo

```python
df = spark.table(DATA_TABLE).toPandas().sort_values("month")
feats = wl.DRIVERS + ["price"]
cut = int(len(df) * 0.8)          # corte em 80 % — sem shuffle
train, test = df.iloc[:cut], df.iloc[cut:]
```

O dataset tem 96 meses ordenados por `month`. A divisão corta em 80 % de
forma cronológica: as primeiras 77 observações viram treino, as últimas 19
viram teste.

!!! note "Conceito — por que sem shuffle em séries temporais?"
    Em um problema de classificação de imagens, a ordem das amostras não
    importa — embaralhar antes de dividir é inofensivo. Em **previsão de
    séries temporais** a ordem importa muito: o modelo aprende a prever
    `price_next_month` a partir dos drivers econômicos do mês atual. Se você
    embaralhar e dividir aleatoriamente, observações futuras podem acabar no
    conjunto de treino — o modelo "vê o futuro" durante o ajuste, e as
    métricas ficam artificialmente infladas. O held-out set deve ser sempre
    as observações mais **recentes**, nunca uma amostra aleatória.

    Esse padrão tem nome: *time-respecting split* ou *walk-forward split*.

!!! tip "Curiosidade — `feats` inclui `price` como feature defasada"
    `wl.DRIVERS` são os dez indicadores econômicos sintéticos
    (`demand_index`, `input_cost_index`, `fx_rate`, `inventory_level`,
    `industrial_output`, `energy_cost`, `scrap_supply`, `export_demand`,
    `seasonality`, `competitor_price`). A eles se soma `price` — o preço
    **atual** da commodity. Essa feature defasada captura momentum: preços
    de commodities tendem a autocorrelacionar-se no curto prazo, então o
    preço de hoje contém informação sobre o preço do próximo mês. É um
    truque simples, mas melhora o R² de forma consistente.

    Repare que a coluna `trend_up` — que indica se o preço subiu — está
    ausente das features. Ela é uma coluna derivada criada em `01_generate_data.py`
    apenas para **ilustração visual**; usá-la seria data leakage direto
    (você calcularia a tendência futura a partir do alvo).

---

## Passo 3 · Treinar e logar um run completo no MLflow

```python
with mlflow.start_run(run_name="sklearn_gbr") as run:
    model = GradientBoostingRegressor(random_state=SEED)
    model.fit(train[feats], train["price_next_month"])
    preds = model.predict(test[feats])

    rmse = float(np.sqrt(mean_squared_error(test["price_next_month"], preds)))
    mae  = float(mean_absolute_error(test["price_next_month"], preds))
    r2   = float(r2_score(test["price_next_month"], preds))

    mlflow.log_params({"model_type": "GradientBoostingRegressor", "seed": SEED})
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

    mlflow.sklearn.log_model(
        model, name="model",
        serialization_format="cloudpickle",
        input_example=test[feats].head(3),
        signature=mlflow.models.infer_signature(test[feats], preds),
    )
    print(f"r2={r2:.3f} rmse={rmse:.2f}")
```

!!! note "Conceito — o que é um MLflow *run*?"
    Um *run* é a unidade atômica de rastreamento no MLflow. Dentro do bloco
    `with mlflow.start_run(...) as run:`, tudo o que você logar fica associado
    a esse run: hiperparâmetros (`log_params`), métricas de avaliação
    (`log_metrics`) e o artefato do modelo (`log_model`). Se o notebook
    abortar no meio do caminho, o run fica marcado como `FAILED` — você não
    corre o risco de ter métricas sem modelo ou vice-versa.

    O `run_name="sklearn_gbr"` é apenas um rótulo legível. O identificador
    real é `run.info.run_id` — um UUID gerado pelo servidor que usaremos no
    próximo passo para registrar o modelo.

!!! note "Conceito — `mlflow.sklearn.log_model` na API MLflow 3.x"
    Na API do MLflow 3.x, o parâmetro correto é `name=`, **não**
    `artifact_path=` (que pertence à API 2.x e está depreciado). `name="model"`
    define o sub-caminho dentro do artefato do run onde o modelo é salvo.
    Essa mudança foi introduzida para alinhar a API de logging com a API
    do registry, onde o modelo também tem um `name`.

    Os outros parâmetros importantes:

    - **`serialization_format="cloudpickle"`** — o formato padrão para
      modelos sklearn no MLflow 3.x é o `skops`, que é mais seguro contra
      desserialização arbitrária mas exige compatibilidade exata de versão
      do scikit-learn. `cloudpickle` é mais portátil entre versões de patch —
      importante num ambiente de workshop onde versões do runtime podem variar
      ligeiramente entre participantes.
    - **`input_example=test[feats].head(3)`** — um exemplo concreto de
      entrada (3 linhas do conjunto de teste). O MLflow usa esse exemplo para
      inferir a assinatura automaticamente *e* para popular a UI de inferência
      em Model Serving. Sem ele, o endpoint de serving fica sem documentação.
    - **`signature=mlflow.models.infer_signature(test[feats], preds)`** —
      a *signature* registra os tipos e shapes esperados de entrada e saída.
      Ela é o contrato formal do modelo: qualquer chamada com tipos errados
      falha cedo, antes de chegar ao `predict`.

!!! tip "Curiosidade — por que `GradientBoostingRegressor` e não XGBoost?"
    GBR do scikit-learn é a escolha deliberada de **minimalismo didático**:
    está disponível em qualquer ambiente Python sem instalação extra, tem
    `random_state` para reprodutibilidade e performa bem no dataset sintético
    (R² esperado ~0,75, dentro da banda [0,6–0,85]). O objetivo do workshop
    não é encontrar o melhor modelo — é mostrar o lifecycle do MLflow. Um
    XGBoost ou LightGBM teria exatamente os mesmos passos de tracking e
    registro; só mudaria o import.

!!! tip "Curiosidade — métricas: RMSE, MAE e R²"
    O notebook loga as três métricas principais de regressão:

    - **RMSE** (Root Mean Squared Error): penaliza erros grandes
      desproporcionalmente (eleva ao quadrado antes de somar). Útil quando
      erros grandes são especialmente custosos — por exemplo, prever preço
      muito abaixo do real leva a compras insuficientes.
    - **MAE** (Mean Absolute Error): a média dos erros absolutos. Mais
      intuitiva ("erro médio de R$ X por unidade") e menos sensível a
      outliers.
    - **R²** (coeficiente de determinação): fração da variância do alvo
      explicada pelo modelo. 0 = não melhor que a média; 1 = perfeito.
      É a métrica do **validation gate** — explicado no próximo passo.

    Os três ficam logados no MLflow e aparecem na UI do experiment para
    comparação entre runs.

---

## Passo 4 · Validation gate — registrar somente se R² ≥ `R2_THRESHOLD`

```python
client = MlflowClient()
if r2 >= R2_THRESHOLD:                                      # 0,6
    mv = mlflow.register_model(
        f"runs:/{run.info.run_id}/model", FORECASTER_MODEL
    )
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate (r2={r2:.3f} >= {R2_THRESHOLD}). "
          f"Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f} < {R2_THRESHOLD}). Not promoted.")
```

`FORECASTER_MODEL` expande para `main.mlops_workshop.price_forecaster`
(definido em `_config.py` como `f"{CATALOG}.{SCHEMA}.price_forecaster"`).

!!! note "Conceito — o que é um validation gate?"
    Um **validation gate** é uma condição explícita e mensurável que um
    modelo precisa satisfazer *antes* de ser promovido para uso downstream.
    Aqui a condição é simples: R² no held-out test ≥ 0,6 (constante
    `R2_THRESHOLD`, definida em `workshop_lib.py` e importada via
    `_config.py`).

    Se o modelo passa:
    1. `mlflow.register_model(...)` cria uma **nova versão** no Unity Catalog
       Registry, copiando o artefato do run para armazenamento versionado.
    2. `set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)`
       move o alias `@champion` para apontar para essa versão.

    Se o modelo falha: o run existe normalmente no experiment (você pode
    inspecionar os parâmetros e métricas para entender por que falhou), mas
    **nenhuma versão de modelo é criada** e nenhum alias se move. O alias
    `@champion` permanece apontando para a última versão aprovada — ou
    simplesmente não existe se nenhuma versão foi aprovada ainda.

!!! note "Conceito — por que um gate determinístico de modelo único, sem champion–challenger?"
    O padrão **champion–challenger** compara dois modelos com tráfego real
    de produção: o champion recebe, digamos, 90 % das requisições; o challenger
    recebe 10 %. Depois de coletar métricas de negócio suficientes, você
    decide qual fica. Esse padrão é poderoso, mas exige um endpoint de
    serving com suporte a divisão de tráfego, um pipeline de coleta de
    feedback, e critérios de promoção baseados em métricas de negócio —
    três peças fora do escopo deste workshop.

    Um limiar determinístico é mais simples de raciocinar e suficiente para
    demonstrar o **princípio central**: promoção é uma decisão, não um
    efeito colateral do treinamento. Enquanto você não decide promover, o
    alias `@champion` não muda — e consumidores downstream continuam usando
    a versão anterior sem interrupção.

!!! note "Conceito — aliases: o contrato estável entre producer e consumers"
    No Unity Catalog Registry, cada versão de modelo tem um número inteiro
    (1, 2, 3…) que cresce a cada novo registro. Re-rodar este notebook
    cria a versão 2; rodar de novo cria a versão 3. Se o Lab 4 ou o Lab 5
    referenciassem `models:/main.mlops_workshop.price_forecaster/1`,
    eles quebrariam silenciosamente assim que uma nova versão melhor fosse
    promovida.

    A solução é o **alias**: um rótulo nomeado que pode ser movido de versão
    para versão sem alterar o código consumidor. `@champion` é o alias
    padrão deste workshop — o contrato entre o pipeline de treino e todo
    downstream. Qualquer código que carregar
    `models:/main.mlops_workshop.price_forecaster@champion` resolve
    automaticamente para a versão aprovada mais recente, hoje e amanhã,
    sem editar nada.

    É o mesmo princípio de um DNS: você não decorou o IP do Google — você
    usa `google.com` e o DNS resolve. O alias é o DNS do modelo.

!!! tip "Curiosidade — versões nunca são deletadas, só aliases se movem"
    Uma vez que uma versão é criada no registry, ela fica lá. Você pode
    arquivar ou deletar versões velhas manualmente, mas o fluxo padrão é
    apenas mover o alias. Isso garante rastreabilidade completa: é sempre
    possível saber qual versão estava em `@champion` em qualquer momento
    no passado (via histórico de aliases na UI ou via `MlflowClient`).

!!! warning "Requer `CREATE MODEL` no schema"
    Se `mlflow.register_model` falhar com erro de permissão, o seu usuário
    ou service principal precisa da permissão `CREATE MODEL` no schema
    `mlops_workshop`. Peça ao administrador do workspace ou execute:

    ```sql
    GRANT CREATE MODEL ON SCHEMA main.mlops_workshop TO `seu-usuario@exemplo.com`;
    ```

    Se o limiar do gate não for atingido, o notebook encerra normalmente
    mas **nenhuma versão é criada**. O Lab 4 (o chain end-to-end) carrega
    `@champion` — portanto você precisa passar o gate antes de avançar.
    Se tentar rodar Lab 4 sem `@champion`, verá uma mensagem de erro clara:
    *"no @champion — run 02_train_forecaster_sklearn and confirm it passed
    the R²≥0.6 gate"*.

!!! success "Pronto quando…"
    - O experiment MLflow em `/Shared/mlops_workshop` contém um run chamado
      `sklearn_gbr` com params, métricas e um artefato de modelo.
    - A célula do notebook imprime `PASSED gate` e um número de versão.
    - `main.mlops_workshop.price_forecaster@champion` resolve para essa
      versão no Catalog Explorer.

---

## Verificar no Catalog Explorer

=== "Saída do notebook"

    Um run bem-sucedido imprime duas linhas:

    ```
    r2=0.xxx rmse=yy.yy
    PASSED gate (r2=0.xxx >= 0.6). Registered v1 as @champion.
    ```

    A partir deste momento, qualquer código que chamar:

    ```python
    mlflow.pyfunc.load_model("models:/main.mlops_workshop.price_forecaster@champion")
    ```

    resolve para essa versão — sem precisar saber o número `v1`.

=== "Catalog Explorer"

    1. Abra **Catalog** na barra lateral esquerda.
    2. Navegue até **main → mlops_workshop → price_forecaster**.
    3. Clique na aba **Versions** — você deve ver a versão 1 com o badge
       do alias `champion`.
    4. A aba **Lineage** vincula de volta ao run do MLflow e à tabela de
       origem `commodity_monthly`.

---

!!! info "📸 Espaço reservado para captura de tela"
    *Capture aqui: o modelo `price_forecaster` registrado com o alias
    `@champion` no Catalog Explorer. Depois substitua por:*
    `![Modelo registrado com alias @champion](../assets/screenshots/lab-2-champion.png)`

---

## Conexão com os próximos labs

O `price_forecaster@champion` que você acabou de criar é o primeiro elo da
cadeia. No **Lab 3**, você vai registrar o otimizador Pyomo — um modelo de
pesquisa operacional sem treinamento — usando o mesmo lifecycle do MLflow,
desta vez com um [PyFunc customizado](../conceitos/pyfunc-modelos-customizados.md).
No **Lab 4**, os dois modelos são carregados via `@champion` e compostos em
uma decisão de compra completa.

**Próximo: [Lab 3 — Registrar o otimizador Pyomo](../lab-3-optimizer/index.md)**
