# 🧩 **Descrição Geral dos Datasets**
Abaixo está uma descrição detalhada dos datasets utilizados no projeto RAG-BR, incluindo sua estrutura, componentes e um dicionário de dados para facilitar a compreensão e manipulação dos mesmos.

## 📚 **Dataset Principal: Quati**
O [**Quati**](https://huggingface.co/datasets/unicamp-dl/quati) é um dataset criado para tarefas de **Recuperação de Informação (IR)** em língua portuguesa, contendo consultas elaboradas por falantes nativos e passagens extraídas de sites brasileiros. Ele é estruturado em três componentes principais:

1. **Passagens** (documentos)  
2. **Consultas** (queries)  
3. **Qrels** (relação consulta–passagem com anotação de relevância)

O dataset está atualmente disponível em duas versões: uma com 1 milhão de passagens (`quati_1M_passages`) e outra maior, com 10 milhões de passagens (`quati_10M_passages`). Até o momento, foram preparados apenas arquivos qrel de validação para ambas as versões, anotando 50 tópicos com uma média de 97,78 passagens por consulta na versão de 10 milhões de passagens e 38,66 passagens por consulta na versão de 1 milhão de passagens.

Algumas alternativas ao Quati podem ser:
- [Megawika](https://huggingface.co/datasets/hltcoe/megawika): Dataset multilingue com trilhões de artigos da Wikipédia em vários idiomas, incluindo português.
- [MFAQ](https://huggingface.co/datasets/clips/mfaq): Dataset de perguntas frequentes em múltiplos idiomas, incluindo português, focado em recuperação de respostas curtas.
- [BeIR](https://huggingface.co/datasets/BeIR/beir): Benchmark de recuperação de informação em inglês com vários datasets no formato ideal para o projeto.

A estrutura do dataset permite avaliar sistemas completos de IR, modelos supervisionados de reranking e pipelines RAG.

---

### 🔎 **Dicionário de Dados**

#### **1. Passagens** (`quati_1M_passages` / `quati_10M_passages`)
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `passage_id` | string | Identificador único da passagem/documento. |
| `passage` | string | Texto completo da passagem em português. |

#### **2. Topics / Consultas** (`quati_all_topics`, `quati_test_topics`, etc.)
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `query_id` | string/int | Identificador único da consulta. |
| `query` | string | Pergunta/consulta formulada por falante nativo. |

#### **3. Qrels — Relevância** (`quati_1M_qrels` / `quati_10M_qrels`)
| Campo | Tipo | Descrição |
|-------|------|-----------|
| `query_id` | string/int | ID da consulta associada. |
| `passage_id` | string | ID da passagem correspondente. |
| `score` | int | Grau de relevância do documento para a consulta (0 ou 3 no dataset). |

---

## 📚 **Dataset para treinamento do reranker: MS MARCO**

O [**MS MARCO (Microsoft MAchine Reading COmprehension)**](https://huggingface.co/datasets/microsoft/ms_marco) é um dos datasets mais utilizados e consolidados para tarefas de **Information Retrieval (IR)**, **Passage Ranking**, **Question Answering** e **treinamento de modelos de reranking supervisionados**.

O dataset foi construído a partir de **consultas reais de usuários do Bing**, associadas a documentos e passagens da web, com anotações humanas indicando quais documentos são relevantes para responder cada consulta. Diferentemente de benchmarks puramente acadêmicos, o MS MARCO reflete **distribuições reais de busca**, com consultas curtas, ambíguas e ruidosas — cenário típico de sistemas de busca e pipelines RAG em produção.

O dataset possui **uma única estrutura de dados**, reutilizada em diferentes **tasks**, e está disponível em **duas versões principais**:

- **v1.1** — maior volume de dados, mais ruído  
- **v2.1** — versão refinada, com menos dados e melhor qualidade

No contexto deste projeto, o MS MARCO é utilizado como **dataset de treinamento do modelo de reranking**, devido à sua **escala**, **qualidade dos rótulos** e **estrutura compatível com modelos Cross-Encoder**, sendo posteriormente aplicado em cenário zero-shot ou cross-lingual para avaliação do Quati no projeto.

---

### 🔎 **Dicionário de Dados**

#### **1. Estrutura Principal**
| Campo | Tipo | Descrição |
|------|------|-----------|
| `query_id` | int32 | Identificador único da consulta. |
| `query` | string | Consulta real formulada por usuário. |
| `query_type` | string | Tipo da consulta (ex.: descrição curta, pergunta, etc.). |
| `answers` | list[string] | Respostas humanas associadas à consulta. |
| `wellFormedAnswers` | list[string] | Respostas bem formadas, usadas principalmente em tarefas de QA. |
| `passages` | dict | Conjunto de passagens candidatas associadas à consulta. |

#### **2. Estrutura Interna de `passages`**
| Campo | Tipo | Descrição |
|------|------|-----------|
| `is_selected` | int32 (0 ou 1) | Indica se a passagem é relevante para a consulta. |
| `passage_text` | string | Texto da passagem/documento. |
| `url` | string | URL do documento de origem. |

---

### 🎯 **Estrutura Implícita de Relevância**

A supervisão de relevância no MS MARCO é **implícita e direta**, definida exclusivamente pelo campo:

- `is_selected = 1` → passagem relevante  
- `is_selected = 0` → passagem irrelevante  

Não há uma tabela `qrels` separada. Cada consulta contém múltiplas passagens candidatas, sendo uma (ou poucas) marcadas como relevantes, refletindo um cenário realista e altamente desbalanceado de recuperação de informação.

Para este projeto, o campo `is_selected` é utilizado como **alvo supervisionado**, sendo convertido para **regressão contínua** (`0.0` / `1.0`) para produção de scores de relevância.
