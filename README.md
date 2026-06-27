# Otimização da Gestão de Acervo em Bibliotecas Universitárias: Uma Abordagem Conjunta de Previsão de Demanda e Recomendação de Livros

> Previsão de demanda e recomendação de livros usando dados reais de empréstimos

---
### Sobre o projeto
 
Este projeto aplica técnicas de **Machine Learning** para otimizar a gestão de acervo em bibliotecas universitárias, abordando dois problemas práticos:
 
1. **Classificação de demanda:** identificar quais livros terão alta demanda de empréstimos, auxiliando na aquisição e reposição do acervo.
2. **Recomendação de livros:** sugerir títulos relevantes com base em similaridade de conteúdo e padrões de empréstimo.
O projeto foi desenvolvido como trabalho final da disciplina de **Aprendizado de Máquina (2026)**, utilizando dados reais de uma biblioteca universitária.

---
### Modelos implementados
 
#### Classificação de Alta Demanda
| Modelo | Abordagem |
|---|---|
| **Random Forest** | Ensemble com GridSearchCV (5-fold CV, otimizado por F1-Score) |
| **XGBoost** | Gradient Boosting com tuning de learning rate, profundidade e subsample |
 
#### Recomendação de Livros (Content-Based)
| Modelo | Abordagem |
|---|---|
| **TF-IDF + Similaridade de Cosseno** | Vetorização de títulos e classificação CDD, top-5 recomendações com métricas de Precisão, Recall e F1 |
| **MiniLM (Sentence Transformers)** | Embeddings semânticos multilingues com `paraphrase-multilingual-MiniLM-L12-v2`, combinando similaridade semântica, estrutural (CDD) e popularidade |
 
---
 
### Estrutura do repositório
 
```
trabalhoML2026/
├── configs/
│   ├── paths.py          # Caminhos dos arquivos de dados
│   └── constants.py      # Hiperparâmetros globais (RANDOM_STATE, TEST_SIZE, limiar de alta demanda)
├── EDA/
│   ├── estatisticas.py   # Cálculo de estatísticas descritivas
│   └── graficos.py       # Visualizações exploratórias
├── dados/
│   ├── carregamento.py   # Leitura e agrupamento dos CSVs de empréstimos
│   └── preprocessamento.py  # Criação de features e variável-alvo
├── modelos/
│   ├── random_forest.py  # Treinamento com GridSearchCV
│   ├── xgboost.py        # Treinamento com GridSearchCV
│   ├── modelo_tfidf.py   # Sistema de recomendação TF-IDF
│   └── minilm.py         # Sistema de recomendação com embeddings MiniLM
├── utils/
│   ├── avaliacao.py      # Métricas e classification report
│   └── visualizacao.py   # Gráficos de feature importance
├── main.ipynb            # Notebook principal (classificação)
├── run_eda.ipynb         # Notebook de análise exploratória
├── MatrizHierarquicaTemporal.csv  # Dataset principal
└── requirements.txt
```
 
---
 
### Como executar
 
**1. Clone o repositório e instale as dependências:**
```bash
git clone https://github.com/alinelimx/trabalhoML2026.git
cd trabalhoML2026
pip install -r requirements.txt
```
 
**2. Configure os caminhos dos dados:**
 
Edite `configs/paths.py` apontando para os seus arquivos CSV de empréstimos.
 
**3. Execute a análise exploratória:**
```bash
jupyter notebook run_eda.ipynb
```
 
**4. Execute o pipeline principal:**
```bash
jupyter notebook main.ipynb
```
 
Para o modelo MiniLM, execute `modelos/minilm.py` no Google Colab (requer GPU para melhor desempenho).
 
---
 
### Tecnologias
 
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-GridSearchCV-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-MiniLM-purple)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)
 
- **Classificação:** `scikit-learn`, `xgboost`
- **NLP / Recomendação:** `sentence-transformers`, `TfidfVectorizer`
- **Análise de dados:** `pandas`, `numpy`
- **Visualização:** `matplotlib`, `seaborn`
---
 
### Pipeline de Dados
 
```
CSVs de empréstimos (2022–2025)
        ↓
Agrupamento por título e classificação CDD
        ↓
Feature Engineering (totais mensais, estatísticas, CV%)
        ↓
Classificação (RF / XGBoost) ──→ Alta Demanda: Sim / Não
        ↓
Recomendação (TF-IDF / MiniLM) ──→ Top-5 títulos similares
```
 