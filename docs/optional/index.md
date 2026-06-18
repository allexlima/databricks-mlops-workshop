# Labs opcionais

Com o caminho obrigatório concluído, você já percorreu o ciclo completo: gerou dados, treinou e registrou dois modelos muito diferentes, compôs a cadeia de decisão e verificou tudo pelo alias `@champion`. Os três notebooks a seguir expandem o que você construiu, cada um em sua própria direção.

Eles são **inteiramente opcionais e independentes entre si**: rode qualquer um, em qualquer ordem, sem afetar os labs principais. Cada um leva entre 10 e 20 minutos.

| Notebook | O que adiciona | Pré-requisito |
|---|---|---|
| `extra/train_forecaster_pytorch.py` | Mesmo lifecycle, framework diferente (PyTorch MLP) | `01_generate_data.py` concluído (tabela de dados existe) |
| `extra/serving.py` | Publica o `price_forecaster@champion` em um endpoint em tempo real | `@champion` definido no `price_forecaster` (`02_train_forecaster_sklearn.py` concluído) |
| `extra/cleanup.py` | Remove todos os recursos criados pelo workshop | Nenhum (mas leia o aviso antes de rodar) |

---

- [Forecaster em PyTorch](pytorch-forecaster.md): mesmo lifecycle, outro framework
- [Model Serving](model-serving.md): publicando o champion em tempo real
- [Limpeza dos recursos](cleanup.md): encerrando o workshop
