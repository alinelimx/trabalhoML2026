# Otimização da Gestão de Acervo em Bibliotecas Universitárias: Uma Abordagem Conjunta de Previsão de Demanda e Recomendação de Livros
Repositório destinado ao trabalho da disciplina de Aprendizado de Máquina
---
### Estrutura
```
trabalhoML2026/
├── configs/
│   ├── paths.py          # caminhos dos arquivos CSV
│   └── constants.py      # RANDOM_STATE, limiar para considerar alta demanda, etc.
├── dados/
│   ├── carregamento.py      # leitura e merge dos CSVs
│   └── preprocessamento.py  # criação de features e target
├── modelos/
│   ├── random_forest.py     
│   └── xgboost.py
├── utils/
│   ├── avaliacao.py     # métricas e classification report
│   └── visualizacao.py  # gráficos de feature importance 
├── main.ipynb            # notebook principal
└── requirements.
```
---
### Como rodar
1. Clone o repositório e instale as dependências: ```pip install -r requirements.txt```
- Ajuste os caminhos dos CSVs em ```config/paths.py``` para o seu ambiente.
- Execute o ```main.ipynb``` do início ao fim.

