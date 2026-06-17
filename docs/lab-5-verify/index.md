# Lab 5 — Verificação (Definition of Done)

Um pequeno harness com quatro asserções que prova que todo o caminho obrigatório funcionou — o seu portão de "rode e veja verde" antes de declarar o workshop concluído.

---

## Antes de começar

!!! warning "Pré-requisito"
    Execute os notebooks `00` a `04` na ordem antes de abrir este lab. O `05_verify.py` não cria nenhum recurso — ele apenas valida o que os labs anteriores produziram.

Abra `05_verify.py`.

---

## Como o harness funciona

O notebook carrega o `MlflowClient`, lê a tabela de features como um DataFrame Pandas ordenado por `month` e percorre quatro checkpoints em sequência. Se qualquer `assert` falhar, a execução para imediatamente e aponta qual checkpoint não passou.

---

## Checkpoint 1 — R² dentro da faixa esperada

```python
r2 = wl.quick_fit_r2(df)
assert wl.R2_BAND[0] <= r2 <= wl.R2_BAND[1], f"R2 {r2:.3f} outside {wl.R2_BAND}"
```

**Por quê?** Antes de tocar o registry, o harness confirma que os dados brutos ainda suportam um ajuste linear razoável. Se o R² estiver fora da faixa `[0,6, 0,85]`, algo mudou nos dados — deriva de distribuição, alteração de schema ou transformação incorreta — e é melhor parar aqui do que promover um modelo treinado em features corrompidas.

---

## Checkpoint 2 — O modelo `price_forecaster` está registrado

```python
assert client.get_registered_model(FORECASTER_MODEL) is not None
```

**Por quê?** Verifica que o nome do modelo existe no Unity Catalog. Falha rápida caso o job de treino (Lab 2) nunca tenha gravado no registry — sem esse registro, os checkpoints seguintes não têm sentido.

---

## Checkpoint 3 — O alias `@champion` resolve

```python
champ = client.get_model_version_by_alias(FORECASTER_MODEL, "champion")
assert champ is not None and champ.version is not None
```

**Por quê?** A asserção é no **alias**, não em um número de versão literal.

!!! info "Por que asserir no alias, não na versão?"
    Cada nova execução do pipeline de treino registra uma versão com número maior. Se o código fixasse `assert champ.version == 3`, ele quebraria na próxima re-execução sem que ninguém tivesse tocado no harness. O alias `@champion` é um ponteiro estável: ele sempre resolve para a versão promovida mais recentemente, independentemente do número que o MLflow atribuiu.

---

## Checkpoint 4 — A cadeia ponta a ponta retorna uma decisão de compra válida

```python
forecaster = mlflow.pyfunc.load_model(f"models:/{FORECASTER_MODEL}@champion")
optimizer  = mlflow.pyfunc.load_model(f"models:/{OPTIMIZER_MODEL}@champion")
latest = df.iloc[[-1]]
pp  = float(forecaster.predict(latest[wl.DRIVERS + ["price"]])[0])
oi  = latest[wl.ECON_COLS].copy(); oi.insert(0, "predicted_price", pp)
dec = optimizer.predict(oi)
assert dec.loc[0, "status"] == "optimal" and pd.notna(dec.loc[0, "purchase_qty"])
```

**Por quê?** Carrega os dois modelos `@champion` e executa um passe completo de inferência na linha mais recente do dataset. O resultado esperado é `status == "optimal"` com um `purchase_qty` não nulo — provando que o forecaster e o optimizer estão compatíveis e que o solver convergiu para uma solução válida. Um `status != "optimal"` geralmente indica uma incompatibilidade entre o intervalo de saída do forecaster e as colunas de orçamento/capacidade na tabela de features.

Ao final do último checkpoint, o notebook imprime:

```
ALL 4 CHECKPOINTS PASSED
```

---

## Referência rápida: passou × falhou

=== "Passou"

    | Checkpoint | O que indica |
    |---|---|
    | R² em `[0,6, 0,85]` | Features dentro do comportamento esperado |
    | Modelo registrado | Lab 2 gravou corretamente no Unity Catalog |
    | `@champion` resolve | Lab 3 ou 4 promoveu uma versão com o alias |
    | `status == "optimal"` | Forecaster e optimizer operam de ponta a ponta |

=== "Se falhar"

    | Checkpoint | Lab para reexecutar |
    |---|---|
    | R² fora da faixa | Verifique `00_setup.py` e a tabela de features (Lab 1) |
    | Modelo não registrado | Reexecute `02_train.py` |
    | `@champion` não resolve | Reexecute `03_evaluate.py` ou `04_promote.py` |
    | `status != "optimal"` | Reexecute `02_train.py` → `03_evaluate.py` → `04_promote.py` |

!!! success "Você deve ver…"
    ```
    ALL 4 CHECKPOINTS PASSED
    ```
    impresso no final do notebook, sem nenhum `AssertionError` nas células anteriores.

---

!!! note "📸 Espaço reservado para captura de tela"
    *Capture aqui: a saída `ALL 4 CHECKPOINTS PASSED`. Depois substitua por `![Verificação aprovada](../assets/screenshots/lab-5-verify.png)`.*

---

Próximo: [Labs opcionais](../optional/index.md)
