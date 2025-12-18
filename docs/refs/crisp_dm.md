# #️⃣ **CRISP-DM — Estrutura Completa do Projeto**

---

# **1. Business Understanding**

### 1.1 Objetivo de Negócio  
Organizações, equipes de pesquisa e analistas lidam com grandes volumes de documentos. Métodos tradicionais de busca por palavra-chave não capturam significado ou contexto.

### 1.2 Problema de Negócio  
Como permitir que um usuário recupere informações profundamente relevantes em uma grande base documental em português e receba uma **resposta fundamentada**, sem depender de leitura manual exaustiva?

### 1.3 Objetivos de Mineração de Dados  
- Construir um sistema de **recuperação semântica eficiente**.  
- Treinar um **modelo supervisionado de reranking**.  
- Integrar recuperação + reranking + geração de resposta (RAG).  
- Expor o pipeline completo via **API**.  

### 1.4 Critérios de Sucesso  
- Recall@K e NDCG significativamente melhores após o reranking.  
- Respostas do LLM fundamentadas nos documentos recuperados.  
- API estável e funcional para uso externo.  
- Pipeline reproduzível e bem documentado.

---

# **2. Data Understanding**

### 2.1 Coleta / Origem dos Dados  
Dataset Quati (HuggingFace):  
https://huggingface.co/datasets/unicamp-dl/quati

### 2.2 Limpeza Inicial dos Dados
- Remover textos com caracteres inválidos.  
- Padronização e normalização básica.  
- Remover blocos ruidosos (HTML, tags).

### 2.3 Descrição dos Dados  
- Consultas escritas por falantes nativos.  
- Documentos de domínio variado (sites em português).  
- Qrels que indicam quais documentos são relevantes para cada consulta.  
- Conjunto adequado para avaliação formal de pipelines de Information Retrieval (IR).

### 2.4 Exploração Inicial  
- Distribuição de tamanhos dos documentos.  
- Exemplos de consultas ambíguas vs diretas.  
- Análise do número médio de documentos relevantes por query.  
- Observação da diversidade de temas.  

### 2.5 Problemas Identificados
- Pouca quantidade de relações entre consultas e documentos.  
- Assimetrias entre positivas e negativas, sendo a maioria dos documentos irrelevantes.  
- Presença de ruído em passagens; necessidade de limpeza cuidadosa.

---

# **3. Data Preparation**

### 3.1 Pré-processamento dos Textos  
- Remoção de HTML e tags.  
- Normalização Unicode.

### 3.2 Tokenização e Chunking
Quebrar passagens muito longas em segmentos de 256–512 tokens.

### 3.3 Embeddings  
Modelos sugeridos:
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`  
- `sentence-transformers/all-MiniLM-L6-v2`  
- `neuralmind/bert-base-portuguese-cased` (convertido para encoder)

### 3.4 Construção do Índice Vetorial  
- FAISS (Flat, HNSW ou IVFFlat dependendo do experimento).
- Armazenar IDs + metadados em VectorDB ou similar.

### 3.5 Dataset para o Reranker
O reranker será treinado com o dataset MS MARCO (em inglês) por sua escala e qualidade de rótulos, e aplicado em zero-shot ao Quati (em português).

A utilização de embeddings multilíngues visa mitigar a barreira linguística, além da aplicação de avaliações qualitativas e métricas de ranking necessárias para validar adaptabilidade cross-linguística entre o MS MARCO e o Quati.

O dataset Quati será utilizado para teste e avaliação do reranker, através das Qrels disponíveis que possuem originalmente valores entre 0 e 3 para score de relevância.

Outra consideração importante é que o modelo de reranking será treinado como um modelo de **regressão**, prevendo scores contínuos de relevância, sendo necessário a normalização dos labels originais para o intervalo [0, 1].

Dessa forma, o modelo aprende a gerar um score contínuo de relevância, permitindo interpretabilidade mais fina e rankings mais expressivos.

A estrutura ideal para os datasets é então: `(query, passage, label)`

---

# **4. Modelling**

### 4.1 Baseline – Recuperação por Embeddings  
Avaliação inicial com FAISS puro:  
- Recall@K  
- NDCG  
- MRR  

### 4.2 Modelo Supervisionado de Reranking

O reranker será um modelo **Cross-Encoder baseado em DistilBERT ou BERTimbau**, treinado como um modelo de regressão, produzindo um score contínuo de relevância entre a consulta e a passagem.

Entrada: `[CLS] query [SEP] passage [SEP]`

Saída: `Score de relevância`, sendo um valor escalar contínuo, entre 0 e 1.

Loss Functions recomendadas:
- Mean Squared Error (MSE) — preferida pela simplicidade e estabilidade.
- Cosine Embedding Loss — opção alternativa quando queremos preservar relações semânticas.
- Pairwise Ranking Loss adaptada para regressão — caso deseje priorizar ordenação.

O uso de regressão permite capturar níveis intermediários de relevância e facilita a inspeção manual do modelo, além de alinhar melhor o uso do score no ranking final.

### 4.3 Pipeline RAG  
Fluxo final:
1. Embedding da query  
2. FAISS → top-k  
3. Reranker → reorder (documentos reordenados em ordem decrescente de relevância)
4. LLM → resposta consolidada  
5. Output estruturado

### 4.4 Modelos de Linguagem para Geração  
- GPT-4o-mini  
- Llama 3 8B Instruct  
- Phi-3-mini  

---

# **5. Evaluation**

### 5.1 Avaliação da Recuperação (pré-reranking)  
- Recall@K  
- Precision@K  
- NDCG@10  
- MRR  

### 5.2 Avaliação do Reranking  
Comparar FAISS vs FAISS + Reranker:

| Métrica | FAISS puro | FAISS + Reranker |
|--------|------------|------------------|
| Recall@5 | X | ↑ Y |
| MRR | X | ↑ Y |
| NDCG@10 | X | ↑ Y |

Como o reranker será um modelo regressivo, que gera scores contínuos, a avaliação do reranking utilizará também métricas de ranking baseadas nesses valores, como:
- Spearman Rank Correlation: correlação entre ranking previsto e ranking ideal
- Kendall Tau: medida de concordância entre rankings
- NDCG@K, agora utilizando diretamente os scores contínuos do modelo

### 5.3 Avaliação do RAG  
- Verificar grounding das respostas.  
- Comparar qualidade com e sem reranking.  
- Avaliações qualitativas e estudos de caso.

### 5.4 Análise de Erros  
- Consultas ambíguas.  
- Passagens muito curtas ou extensas.  
- Reranking invertendo documentos importantes.

---

# **6. Deployment**

### 6.1 API (FastAPI)  
Endpoints sugeridos:

- Para usuário final:
```yaml
POST /query
  body: { "query": "...", "k": 5 }
  returns:
    - documentos recuperados (IDs, títulos, scores)
    - resposta síntese gerada pelo LLM
    - scores do reranker
```

- Para uso interno / testes:
```yaml
POST /embed
POST /rerank
POST /rag
```

### 6.2 Docker  
- Dockerfile (baseado em Python 3.10+)
- requirements.txt com dependências (a revisar):
    - fastapi
    - uvicorn
    - sentence-transformers
    - faiss
    - transformers
    - openai API ou huggingface
    - torch
- Uvicorn server  

### 6.3 Scripts Essenciais Executáveis
- `ingest.py` – para ingestão e indexação; constrói embeddings, chunks e índice FAISS  
- `train_reranker.py` – treina o modelo supervisionado  
- `evaluate.py` – executa todas as métricas  
- `run_api.py` – roda a API completa

### 6.4 Instruções de Uso
- Como rodar a API
- Como treinar o reranker
- Como usar a aplicação como usuário final

<!-- ### 6.6 Interface básica via Streamlit (opcional)
- Interface simples para testes manuais -->

<!-- ### 6.7 Monitoramento e Logs
- Métricas de latência e throughput da API  
- Logs de requisições e erros via FastAPI middleware  
- Logs de erros e exceções  
- Monitoramento básico de performance (tempo de resposta, uso de memória)
- Alertas para falhas críticas -->

---

# **7. Conclusões**

### 7.1 Síntese dos Resultados  
- Embeddings geram uma boa recuperação inicial.  
- O reranker supervisionado melhora significativamente a qualidade final.  
- O pipeline RAG produz respostas fundamentadas e coerentes.  

### 7.2 Lições Aprendidas Esperadas  
- Estruturação do trabalho via CRISP-DM.  
- Dados ruidosos exigem preparação cuidadosa.  
- Rerankers leves têm excelente custo-benefício.  

### 7.3 Trabalhos Futuros  
- Fine-tuning de embeddings customizados.  
- Ampliação da base de documentos.  
- Avaliação humana sistematizada.

---

# **8. Appendix**

- Exemplos de chamadas à API  
- Amostras do dataset  
- Tabelas de métricas adicionais  
- Links para notebooks de desenvolvimento