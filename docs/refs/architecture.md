# 🧱 Arquitetura do Sistema — RAG-BR

Este documento descreve a arquitetura completa do projeto **RAG-BR**, contemplando ingestão, indexação, embeddings, treinamento do modelo de reranking, pipeline RAG e exposição via API.  
O objetivo é fornecer uma visão clara, modular e reprodutível da solução, alinhada às práticas modernas de MLOps e engenharia de IA.

---

# 1. Visão Geral da Arquitetura

A arquitetura é composta por cinco grandes módulos:

1. **Ingestão e Pré-processamento**
2. **Construção de Embeddings e Indexação Vetorial (FAISS)**
3. **Treinamento do Reranker Supervisionado**
4. **Pipeline RAG (Retriever → Reranker → Generator)**
5. **API de Exposição (FastAPI + Docker)**

O fluxo completo:

```
Datasets (Quati, MS MARCO)
  ↓
[Pré-processamento: limpeza, normalização, chunking]
  ↓
Chunks + Metadados
  ↓
[Geração de Embeddings]
  ↓
FAISS Index + Embeddings
  ↓
[Treinamento do Reranker (Cross-Encoder)]
  ↓
Reranker Supervisionado
  ↓
[Pipeline RAG: Query → Retrieval → Reranking → Generation]
  ↓
API (FastAPI + Docker)
  ↓
Resposta Estruturada (texto + evidências + scores)
```

---

# 2. Módulo de Ingestão e Pré-processamento

Responsável por:

- Carregar os datasets Quati e MS MARCO da HuggingFace.
- Remover ruídos (HTML, caracteres inválidos).
- Normalizar textos e labels.
- Salvar artefatos processados em formato Parquet.

### Artefatos gerados:
- `data/processed/quati_reranker_eval.parquet`
- `data/processed/msmarco_reranker_train.parquet`

---

# 3. Módulo de Embeddings e Indexação Vetorial

Este módulo gera e armazena representações vetoriais:

- Modelos recomendados:
  - `paraphrase-multilingual-mpnet-base-v2`
  - `all-MiniLM-L6-v2`

Passos:

1. Gerar embeddings dos chunks.
2. Construir índice FAISS (HNSW ou IVFFlat).
3. Persistir índice e metadados.

### Artefatos gerados:
- `embeddings.npy`
- `faiss.index`
- `faiss_metadata.parquet`

---

# 4. Treinamento do Modelo de Reranking

O reranker é um **Cross-Encoder** treinado como **regressor**, produzindo scores contínuos de relevância.

### Entrada:
`[CLS] query [SEP] passage [SEP]`

### Saída:
`score_contínuo ∈ [0, 1]`

Utiliza Qrels do Quati para supervisionar.

### Artefatos gerados:
- `models/reranker/`
- `training_metrics.json`

---

# 5. Pipeline RAG

Fluxo:

1. **Embedding da query**
2. **Recuperação inicial via FAISS** (top-K)
3. **Reranking supervisionado** (ordena por relevância)
4. **Geração da resposta via LLM**, usando somente documentos reranqueados
5. **Retorno estruturado** contendo:
   - resposta final
   - documentos usados
   - scores e ranking
   - evidências

---

# 6. API (FastAPI)

Endpoints principais:

```yaml
POST /query Executa o pipeline RAG completo
POST /embed Gera embeddings
POST /rerank Aplica o reranker
POST /rag Executa RAG sem resposta sintetizada
```

A API é empacotada com Docker e servida com Uvicorn.

---

# 7. Armazenamento e Organização dos Artefatos

```
data/
raw/
processed/
index/
models/
reranker/
embeddings/
```

---

# 8. Futuras Extensões

- Cache de embeddings e respostas
- Monitoramento de latência e throughput
- Fine-tuning de embeddings
- Avaliação humana sistemática

---

# Conclusão

Esta arquitetura equilibra:
- precisão,
- escalabilidade,
- rastreabilidade,
- modularidade.

Servirá como mapa de referência para toda a implementação do projeto.
