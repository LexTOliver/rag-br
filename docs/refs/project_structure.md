# 📁 Estrutura do Projeto — RAG-BR

Este documento descreve a estrutura oficial prevista de diretórios e arquivos do projeto, servindo como referência para implementação, organização e manutenção.

---

# 1. Estrutura Geral

```yaml
rab-br/
│
├── data/
│ ├── raw/ # Dados brutos
│ ├── processed/ # Dados limpos/chunkados/curados
│ └── index/ # Índice FAISS + metadados
│
├── models/
│ ├── reranker/ # Modelo treinado
│ ├── embeddings/ # Modelos de embeddings
│ └── llm_cache/ # Cache opcional
│
├── docs/
│ ├── study_reports/ # Relatórios para entrega de atividades
│ └── refs/ # Documentação técnica
│
├── notebooks/ # Jupyter Notebooks
│ ├── 01_data_understanding.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_embeddings_index.ipynb
│ ├── 04_train_reranker.ipynb
│ ├── 05_evaluation.ipynb
│ └── 06_rag_pipeline_tests.ipynb
│
├── src/
│ ├── ingest/
│ │ ├── load_dataset.py
│ │ ├── preprocess.py
│ │ ├── chunking.py
│ │ ├── embed.py
│ │ └── build_index.py
│ │
│ ├── training/
│ │ ├── dataset_builder.py
│ │ ├── train_reranker.py
│ │ └── evaluate_reranker.py
│ │
│ ├── rag/
│ │ ├── retriever.py
│ │ ├── reranker.py
│ │ ├── generator.py
│ │ └── pipeline.py
│ │
│ ├── api/
│ │ ├── main.py
│ │ ├── schemas.py
│ │ └── controllers.py
│ │
│ ├── utils/
│ │ ├── io.py
│ │ ├── logging.py
│ │ └── config.py
│ │
│ └── tests/
│ ├── test_index.py
│ ├── test_reranker.py
│ └── test_api.py
│
├── configs/
│ ├── embed_config.yaml
│ ├── index_config.yaml
│ ├── training_config.yaml
│ └── api_config.yaml
│
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```


---

# 2. Guia de Propósito por Pasta

### `data/`
Armazena dados **não versionados** (`.gitignore`).  
Separa dados brutos, processados e índices.

### `models/`
Armazena modelos treinados, checkpoints e metadados.

### `docs/`
Documentação técnica modular e relatórios do projeto de estudo.

### `notebooks/`
Ambiente exploratório.  
Ordem numerada reflete o fluxo CRISP-DM e MLOps.

### `src/`
Código de produção, organizado por domínio lógico, contendo:
- `ingest/`: Ingestão e pré-processamento de dados.
- `training/`: Treinamento e avaliação do modelo de reranking.
- `rag/`: Implementação do pipeline RAG (recuperação, reranking, geração).
- `api/`: Código da API FastAPI para deploy.
- `utils/`: Funções utilitárias reutilizáveis.
- `tests/`: Testes unitários e de integração. 

### `configs/`
Hiperparâmetros, caminhos, parâmetros de indexação, etc.

### `Makefile`
Automatiza tarefas:

```yaml
make ingest
make index
make train
make api
```

---

# 3. Filosofia da Estrutura

Esta organização segue boas práticas usadas em:

- projetos de MLOps industriais,
- pipelines RAG de larga escala,
- ambientes corporativos com CI/CD,
- projetos acadêmicos robustos.

Forte separação entre:

- **exploração (notebooks)**  
- **produção (src)**  
- **documentação (docs)**  
- **artefatos (data/models)**  

---

# Conclusão

Esta estrutura é modular, escalável e facilmente navegável.  
Ela serve como referência oficial para desenvolvimento e extensão do projeto RAG-BR.
