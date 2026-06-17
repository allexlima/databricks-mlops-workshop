# Lab 1 — Gerar o conjunto de dados

Dados reprodutíveis e governados são a base de qualquer ciclo de vida de modelos — sem uma única fonte de verdade, comparar o desempenho de modelos diferentes é como comparar maçãs com laranjas. Neste lab você cria o dataset sintético compartilhado que todos os outros labs consomem.

Abra `01_generate_data.py`.

---

## Passo 1 · Carregar a configuração e gerar o DataFrame bruto

```python
%run ./_config          # define CATALOG, SCHEMA, DATA_TABLE, CSV_PATH, SEED

import workshop_lib as wl

df = wl.generate_dataset(seed=SEED)   # 96 linhas mensais, ordenadas no tempo
display(df.head(10))
```

O `%run ./_config` centraliza todas as constantes de workspace (catálogo, esquema, caminho do Volume) em um único arquivo — você edita ali e todos os notebooks herdam a mudança sem copiar e colar. `generate_dataset` devolve um **pandas DataFrame com 96 linhas**, uma por mês, construído com random-walks cumulativas e sazonalidade senoidal; as linhas **nunca são embaralhadas**, o que mantém os cortes temporais honestos para avaliação de séries temporais.

!!! info "Por que dados sintéticos?"
    Um gerador determinístico compartilhado garante que todos os participantes treinem e avaliem sobre exatamente os mesmos dados. Nenhum dado real de cliente é manipulado no ambiente do workshop, e a estrutura causal é próxima o suficiente de um mercado de commodities real para que os problemas de ML sejam não-triviais.

---

## Passo 2 · Dicionário de dados e verificações de sanidade

```python
print(df.describe())
print("\nCorrelação dos drivers com o alvo:")
print(df[wl.DRIVERS].corrwith(df["price_next_month"]).sort_values())
```

O dataset tem **10 features de driver** que cobrem dinâmicas de demanda, custo, macro e oferta:

| Feature | Papel |
|---|---|
| `demand_index` | Índice de demanda do mercado |
| `input_cost_index` | Custo de insumos |
| `fx_rate` | Taxa de câmbio |
| `inventory_level` | Nível de estoque |
| `industrial_output` | Produção industrial |
| `energy_cost` | Custo de energia |
| `scrap_supply` | Oferta de sucata |
| `export_demand` | Demanda de exportação |
| `seasonality` | Componente sazonal senoidal |
| `competitor_price` | Preço do concorrente |

A coluna alvo `price_next_month` é uma função ponderada dos mesmos drivers + preço defasado + ruído, então as correlações são não-nulas mas não perfeitamente lineares — o sinal é **aprendível mas não trivial**.

Além dos drivers, cinco colunas econômicas (`holding_cost`, `purchase_cost`, `demand`, `capacity`, `budget`) alimentam o modelo de otimização Pyomo nos labs seguintes.

!!! note "📸 Espaço reservado para captura de tela"
    *Capture aqui: a saída do dicionário de dados / verificações de sanidade. Depois substitua este bloco por `![Dicionário de dados](../assets/screenshots/lab-1-dados.png)`.*

---

## Passo 3 · Baseline ingênuo vs. R² do modelo rápido

```python
from sklearn.metrics import r2_score

cut = int(len(df) * 0.8)
naive_r2 = r2_score(df["price_next_month"].iloc[cut:], df["price"].iloc[cut:])
model_r2 = wl.quick_fit_r2(df)   # corte temporal 80/20, GBR pequeno
print(f"naive R2={naive_r2:.3f}  quick-model R2={model_r2:.3f}")
```

O **forecast ingênuo lag-1** (usar o preço de hoje como previsão para amanhã) define o **piso** — qualquer modelo que não consiga superar esse R² não aprendeu nada útil. `quick_fit_r2` treina um Gradient Boosting Regressor pequeno com um corte temporal estrito de 80/20 para definir o **teto**. Todo forecaster que você construir nos labs seguintes deve ficar **entre** esses dois valores.

!!! info "Por que um corte temporal — e não aleatório?"
    Em séries temporais, um split aleatório vaza informação do futuro para o treino (o modelo vê dados posteriores ao período de teste). O corte temporal preserva a causalidade: o modelo só treina no passado e é avaliado no futuro.

---

## Passo 4 · Portão de qualidade do sinal

```python
lo, hi = wl.R2_BAND    # (0.6, 0.85)
assert lo <= model_r2 <= hi, f"R2 {model_r2:.3f} outside band {wl.R2_BAND}"
```

Este `assert` é um **checkpoint de qualidade dos dados**, não um teste de modelo. Ele verifica que o gerador está calibrado corretamente: um R² abaixo de 0,6 significa que o sinal é ruidoso demais para qualquer modelo aprender; acima de 0,85 o problema é trivialmente fácil e nenhum modelo terá dificuldades reais para ser avaliado. Se o assert disparar, o problema está nos parâmetros de ruído de `workshop_lib.generate_dataset`, não no seu código de modelo.

!!! warning "Se o assert falhar"
    Uma mudança de seed ou uma versão diferente do NumPy pode deslocar os valores gerados o suficiente para mover o R² para fora da faixa. Verifique que `SEED = 42` está em `_config`, confirme que as versões de `numpy` e `scikit-learn` batem com o ambiente fixado (`databricks_ml_v5`) e re-execute a célula.

---

## Passo 5 · Persistir como tabela Delta + CSV em um UC Volume

```python
spark.createDataFrame(df).write.mode("overwrite").saveAsTable(DATA_TABLE)
df.to_csv(CSV_PATH, index=False)
print(f"Wrote {DATA_TABLE} and {CSV_PATH}.")
```

Dois formatos, dois propósitos:

| Destino | Finalidade |
|---|---|
| Tabela Delta no Unity Catalog | Governada, consultável por SQL, com trilha de auditoria — a fonte canônica para todos os labs |
| CSV em UC Volume | Portátil; notebooks que precisam ler dados antes de um contexto Spark disponível usam este arquivo diretamente |

!!! warning "Use Volumes, não `/dbfs/`, em compute serverless"
    Clusters serverless não montam `/dbfs/`. Sempre escreva arquivos portáteis em um UC Volume (`/Volumes/<catalog>/<schema>/<volume>/…`). O `CSV_PATH` em `_config` já faz isso — não mude para um caminho `/dbfs/`.

---

## Inspecionar a tabela escrita

=== "Python"

    ```python
    display(spark.table(DATA_TABLE))     # DATA_TABLE vem de _config
    spark.table(DATA_TABLE).count()      # deve ser 96
    ```

=== "SQL"

    ```sql
    SELECT count(*) FROM main.mlops_workshop.commodity_monthly;
    -- Esperado: 96
    SELECT * FROM main.mlops_workshop.commodity_monthly LIMIT 5;
    ```

!!! success "Checkpoint — você deve ver"
    - **96 linhas** na tabela Delta (uma por mês, coluna `month` de 0 a 95).
    - A célula do `assert` passa silenciosamente (R² impresso, sem `AssertionError`).
    - Mensagem `Wrote main.mlops_workshop.commodity_monthly and /Volumes/…` impressa na última célula.

---

Próximo: [Lab 2 — Treinar o forecaster](../lab-2-forecaster/index.md)
