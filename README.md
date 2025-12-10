# **RAG-BR: Sistema de Recuperação, Reranking e Resposta Baseada em Documentos em Português com Modelo de Reranking Treinado**
## **Aplicando CRISP-DM para um problema de negócio**

- **Trabalho Final de Pós-Graduação**
- **Autor:** Alexandre Oliveira
- **RA:** 52400856

---

## 🎯 **Objetivo do Trabalho**

Este trabalho implementa um pipeline completo de **Recuperação Aumentada por Geração (RAG)** para textos em português, integrando **embeddings semânticos**, **indexação vetorial**, um **modelo de reranking treinado pelo autor** e um módulo de **resposta-resumo baseado em LLM**.

O sistema retorna documentos relevantes da base, apresenta o conjunto de evidências utilizado e gera uma resposta fundamentada, combinando técnicas modernas de PLN, engenharia de modelos e princípios de MLOps. Todo o projeto segue rigorosamente a metodologia **CRISP-DM**, desde o entendimento do problema até o deploy final como uma API.

---

## 📚 Documentação Técnica
A documentação detalhada do projeto está disponível na pasta `docs/` (a revisar):

- [Relatórios de Estudo](./docs/study_reports/)
- [Arquitetura do Sistema](./docs/refs/architecture.md)
- [Estrutura do Projeto](./docs/refs/project_structure.md)
<!-- TODO: Mover documentação da metodologia CRISP-DM para docs/ -->
<!-- TODO: Adicionar documento de Referências dos Notebooks do Google Colab -->
<!-- TODO: Mover descrição do dataset para docs/ -->
<!-- TODO: Adicionar documentação do pipeline RAG -->
<!-- TODO: Adicionar documentação sobre treinamento do Reranker -->

Esses documentos servem como guia técnico durante toda a implementação.

---

# 🧩 **Descrição Geral do Quati Dataset**

O [**Quati**](https://huggingface.co/datasets/unicamp-dl/quati) é um dataset criado para tarefas de **Recuperação de Informação (IR)** em língua portuguesa, contendo consultas elaboradas por falantes nativos e passagens extraídas de sites brasileiros. Ele é estruturado em três componentes principais:

1. **Passagens** (documentos)  
2. **Consultas** (queries)  
3. **Qrels** (relação consulta–passagem com anotação de relevância)

O dataset está atualmente disponível em duas versões: uma com 1 milhão de passagens (`quati_1M_passages`) e outra maior, com 10 milhões de passagens (`quati_10M_passages`). Até o momento, foram preparados apenas arquivos qrel de validação para ambas as versões, anotando 50 tópicos com uma média de 97,78 passagens por consulta na versão de 10 milhões de passagens e 38,66 passagens por consulta na versão de 1 milhão de passagens.

Essa estrutura permite treinar e avaliar sistemas completos de IR, modelos supervisionados de reranking e pipelines RAG.

---

## 🔎 **Dicionário de Dados**

### **1. Passagens** (`quati_1M_passages` / `quati_10M_passages`)
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `passage_id` | string | Identificador único da passagem/documento. |
| `passage` | string | Texto completo da passagem em português. |

### **2. Topics / Consultas** (`quati_all_topics`, `quati_test_topics`, etc.)
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `query_id` | string/int | Identificador único da consulta. |
| `query` | string | Pergunta/consulta formulada por falante nativo. |

### **3. Qrels — Relevância** (`quati_1M_qrels` / `quati_10M_qrels`)
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `query_id` | string/int | ID da consulta associada. |
| `passage_id` | string | ID da passagem correspondente. |
| `score` | int | Grau de relevância do documento para a consulta (0 ou 1 no dataset). |

---

# 🏗️ **Visão Geral do Pipeline Proposto**

1. Usuário envia **uma pergunta** ou **um documento**.  
2. Sistema gera um **embedding semântico**.  
3. Busca inicial dos documentos mais similares via **FAISS** (top-k).  
4. Documentos são reordenados pelo **modelo de reranking treinado** com pares do Quati.  
5. Os documentos reranqueados e reordenados são passados para um **LLM** para geração de resposta.  
6. O sistema retorna:  
   - resposta fundamentada,  
   - lista dos documentos utilizados,  
   - scores de relevância,  
   - ranking antes e depois do reranking.  
7. Deploy final via **FastAPI + Docker**.

---

# #️⃣ **CRISP-DM — Estrutura Completa do Projeto**

---

# **1. Business Understanding**

### 1.1 Objetivo de Negócio  
Organizações, equipes de pesquisa e analistas lidam com grandes volumes de documentos. Métodos tradicionais de busca por palavra-chave não capturam significado ou contexto.

### 1.2 Problema de Negócio  
Como permitir que um usuário recupere informações profundamente relevantes em uma grande base documental em português e receba uma **resposta fundamentada**, sem depender de leitura manual exaustiva?

### 1.3 Objetivos de Mineração de Dados  
- Construir um sistema de **recuperação semântica eficiente**.  
- Treinar um **modelo supervisionado de reranking** usando o Quati.  
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

### 2.2 Descrição dos Dados  
- Consultas escritas por falantes nativos.  
- Passagens de domínio variado (sites em português).  
- Qrels que indicam quais passagens são relevantes para cada consulta.  
- Conjunto adequado para avaliação formal de pipelines de Information Retrieval (IR).

### 2.3 Exploração Inicial (esperada no notebook)  
- Distribuição de tamanhos das passagens.  
- Exemplos de consultas ambíguas vs diretas.  
- Análise do número médio de documentos relevantes por query.  
- Observação da diversidade de temas.  

### 2.4 Problemas Possíveis  
- Documentos extensos precisam de **chunking**.  
- Algumas consultas são curtas demais.  
- Assimetrias entre positivas e negativas.  
- Variação grande de estilo linguístico.

---

# **3. Data Preparation**

### 3.1 Limpeza
- Remover textos com caracteres inválidos.  
- Padronização e normalização básica.  
- Remover blocos ruidosos (HTML, tags).

### 3.2 Tokenização / Embeddings  
Modelos sugeridos:
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`  
- `sentence-transformers/all-MiniLM-L6-v2`  
- `neuralmind/bert-base-portuguese-cased` (convertido para encoder)

### 3.3 Chunking  
Quebrar passagens muito longas em segmentos de 256–512 tokens.

### 3.4 Construção do Índice Vetorial  
- FAISS (Flat, HNSW ou IVFFlat dependendo do experimento).
- Armazenar IDs + metadados.

### 3.5 Dataset para o Reranker  

O dataset de reranking será preparado utilizando os Qrels, onde score é originalmente binário (0 ou 1), mas será tratado como um alvo contínuo em uma tarefa de regressão.

Dessa forma, o modelo aprende a gerar um score contínuo de relevância, permitindo interpretabilidade mais fina e rankings mais expressivos.

Montado a partir dos Qrels: `(query, passage, label)`

>Obs.:
  label = 1 se score=1 em `qrels`;
  label = 0 se score=0

---

# **4. Modeling**

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