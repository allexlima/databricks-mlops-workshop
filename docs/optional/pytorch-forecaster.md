# Forecaster em PyTorch: mesmo lifecycle, outro framework

Este lab mostra que o MLflow governa qualquer modelo Python, não apenas scikit-learn: basta trocar o objeto do modelo e o restante do ciclo permanece idêntico.

Abra `extra/train_forecaster_pytorch.py`.

---

## Por que este lab existe

O `02_train_forecaster_sklearn.py` treina um gradient boosting com scikit-learn. Este interlude substitui esse modelo por um pequeno MLP (Multi-Layer Perceptron) em PyTorch. O objetivo não é comparar desempenho entre os dois frameworks, e o PyTorch não é um challenger ao sklearn: ambos registram no mesmo modelo `FORECASTER_MODEL` e disputam o mesmo alias `@champion`. A mensagem é outra: **o MLflow governa o que você tiver** porque o mecanismo de tracking, registro e alias opera no nível do run e da versão, independente do framework que gerou o artefato.

!!! note "Conceito"
    Um framework de deep learning como PyTorch expõe seus modelos como objetos Python comuns (`nn.Module`). O MLflow tem um flavor nativo, `mlflow.pytorch`, que serializa e desserializa esses objetos automaticamente, incluindo os pesos treinados. Na hora de carregar o modelo, você recebe de volta um `nn.Module` funcional, sem precisar reescrever nenhuma classe.

---

## O modelo: MLP de duas camadas

O notebook define um pequeno perceptron multicamada totalmente conectado: duas camadas lineares separadas por uma ativação ReLU, com 32 neurônios na camada oculta. A arquitetura, uma subclasse de [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), é deliberadamente simples: o foco é o lifecycle, não a arquitetura.

!!! example "Cole no notebook (parte 1 de 2)"
    Esta célula recebe **dois blocos**. Cole primeiro a classe `MLP` abaixo no
    espaço reservado da célula **«O modelo + Tracking»**; o bloco de treino do
    próximo passo vai logo em seguida, na mesma célula.

```python
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)
```

O treinamento usa Adam por 400 épocas com `lr=0.01` e MSE como função de perda. A semente aleatória é fixada via `torch.manual_seed(SEED)`, o mesmo `SEED` do `_config.py` que todos os notebooks herdam, para garantir reprodutibilidade entre execuções.

---

## Dados, split e normalização: idênticos ao forecaster sklearn

O notebook lê a mesma tabela Delta criada em `01_generate_data.py` (`DATA_TABLE`), aplica o mesmo corte cronológico 80/20, e normaliza as features com z-score calculado **exclusivamente sobre os dados de treino**. Por fim, converte os arrays em tensores PyTorch (`Xtr`, `ytr`, `Xte`) prontos para o treino.

!!! example "Cole no notebook"
    Substitua o espaço reservado da célula **«Dados, split e normalização»** pelo bloco abaixo.

```python
df = spark.table(DATA_TABLE).toPandas().sort_values("month")
feats = wl.DRIVERS + ["price"]
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]

mu, sd = train[feats].mean(), train[feats].std(ddof=0)

def to_t(frame):
    return torch.tensor(((frame[feats] - mu) / sd).to_numpy(), dtype=torch.float32)

Xtr = to_t(train)
ytr = torch.tensor(train["price_next_month"].to_numpy(), dtype=torch.float32).view(-1, 1)
Xte = to_t(test)
```

!!! tip "Curiosidade"
    Usar a média e o desvio padrão do conjunto de treino para normalizar o conjunto de teste é um requisito fundamental: se você calcular essas estatísticas sobre o conjunto de teste (ou sobre o dataset inteiro), o modelo "vê" informações do futuro durante o treino, o que é data leakage. Normalizar antes do treino é um dos ajustes de maior impacto para estabilizar a convergência de MLPs.

---

## Tracking e log: dentro de um único `mlflow.start_run`

Todo o treinamento acontece dentro de um bloco [`mlflow.start_run`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run) (`run_name="pytorch_mlp"`). Isso garante que parâmetros, métricas e o artefato do modelo pertençam à mesma run, e que o `info.model_uri` devolvido por `log_model` seja a URI correta para o passo de registro.

!!! example "Cole no notebook (parte 2 de 2)"
    Cole este bloco **logo após a classe `MLP`**, na mesma célula
    **«O modelo + Tracking»**.

```python
with mlflow.start_run(run_name="pytorch_mlp"):
    net = MLP(len(feats))
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(400):
        opt.zero_grad()
        loss = loss_fn(net(Xtr), ytr)
        loss.backward()
        opt.step()

    preds = net(Xte).detach().numpy().ravel()
    r2 = float(r2_score(test["price_next_month"], preds))
    rmse = float(np.sqrt(mean_squared_error(test["price_next_month"], preds)))

    mlflow.log_params({"model_type": "torch_mlp", "epochs": 400, "seed": SEED})
    mlflow.log_metrics({"r2": r2, "rmse": rmse})
    info = mlflow.pytorch.log_model(net, name="model")
    print(f"r2={r2:.3f} rmse={rmse:.2f}")
```

O experimento é o mesmo que o `02_train_forecaster_sklearn.py` usa, definido em `EXPERIMENT_PATH` no `_config.py`. Isso significa que as runs do sklearn e do PyTorch aparecem **lado a lado** na mesma view do MLflow, facilitando a comparação de métricas sem sair da interface.

!!! note "Conceito"
    A API `mlflow.pytorch.log_model(net, name="model")` é a forma MLflow 3.x de registrar o artefato: o parâmetro `name=` substitui o antigo `artifact_path=` (depreciado). O objeto serializado é um checkpoint PyTorch padrão (`.pt`), e o MLflow armazena junto os metadados do modelo para que o flavor correto seja usado na hora do carregamento.

---

## Validation gate e promoção para `@champion`

O mesmo `R2_THRESHOLD` (0.6) do forecaster sklearn. Se o modelo passar, [`mlflow.register_model`](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.register_model) o registra no Unity Catalog com o mesmo nome `FORECASTER_MODEL` e `set_registered_model_alias` aplica o alias `@champion`, podendo sobrescrever a versão sklearn anterior.

!!! example "Cole no notebook"
    Substitua o espaço reservado da célula **«Validation gate»** pelo bloco abaixo.

```python
client = MlflowClient()
if r2 >= R2_THRESHOLD:
    mv = mlflow.register_model(info.model_uri, FORECASTER_MODEL)
    client.set_registered_model_alias(FORECASTER_MODEL, "champion", mv.version)
    print(f"PASSED gate. Registered v{mv.version} as @champion.")
else:
    print(f"FAILED gate (r2={r2:.3f}). Not promoted.")
```

Isso é intencional: diferentes frameworks competindo pelo mesmo alias de produção. O `04_end_to_end.py` carrega o `@champion` pelo alias (nunca pelo número de versão), então funciona independentemente de qual framework está apontado.

!!! note "Conceito"
    Este é o poder dos aliases no Unity Catalog: desacoplam o código consumidor da versão concreta. O `04_end_to_end.py` sempre chama `client.get_model_version_by_alias(FORECASTER_MODEL, "champion")`, nunca `v3` ou `v7`. Quem promoveu qual versão é detalhe de governança, não de código. Ver mais em [Carregar os modelos por @champion](../lab-4-end-to-end/carregar-modelos.md).

---

## Dependência de compute: `torch` no bloco PEP 723

O notebook declara `torch` como dependência extra no bloco PEP 723, que é a única diferença de ambiente em relação ao forecaster sklearn. O torch é configurado para download CPU-only para evitar o peso das wheels CUDA.

```python
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

!!! success "Pronto quando..."
    - A saída mostra `PASSED gate. Registered v<N> as @champion.` (ou `FAILED gate (r2=...)`, aceitável se o ambiente for menos estável; tente aumentar as épocas).
    - Uma run chamada `pytorch_mlp` aparece no experimento MLflow compartilhado.
    - O Unity Catalog mostra `FORECASTER_MODEL@champion` apontando para a nova versão PyTorch.

---

**Próximo passo (opcional):** [Model Serving](model-serving.md)
