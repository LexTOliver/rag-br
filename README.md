# **RAG-BR: Sistema de Recuperação, Reranking e Resposta Baseada em Documentos em Português com Modelo de Reranking Treinado**
## **Aplicando CRISP-DM para um problema de negócio**

- **Trabalho Final de Pós-Graduação**
- **Especialização em Data Science e Machine Learning**
- **Autor:** Alexandre Oliveira

---

## 🎯 **Objetivo do Trabalho**

Este trabalho implementa um pipeline completo de **Recuperação Aumentada por Geração (RAG)** para textos em português, integrando **embeddings semânticos**, **indexação vetorial**, um **modelo de reranking treinado pelo autor** e um módulo de **resposta-resumo baseado em LLM**.

O sistema retorna documentos relevantes da base, apresenta o conjunto de evidências utilizado e gera uma resposta fundamentada, combinando técnicas modernas de PLN, engenharia de modelos e princípios de MLOps. Todo o projeto segue rigorosamente a metodologia **CRISP-DM**, desde o entendimento do problema até o deploy final como uma API.

---

## 🚀 **Status do Projeto**
- ✔️ Análise Exploratória
- 🔄 Data Preparation (limpeza, normalização, seleção de features)
- 🔄 Embeddings e Indexação Vetorial
- ⏳ Treinamento de Reranker  
- ⏳ Pipeline RAG
- ⏳ Deploy via FastAPI + Docker

---

## 📚 Documentação Técnica
A documentação detalhada do projeto está disponível na pasta `docs/` (a revisar):

- [Relatórios de Estudo](./docs/study_reports/)
- [Metodologia CRISP-DM](./docs/refs/crisp_dm.md)
- [Arquitetura do Sistema](./docs/refs/architecture.md)
- [Estrutura do Projeto](./docs/refs/project_structure.md)
- [Descrição dos Datasets](./docs/refs/datasets_description.md)
- [Google Colab Notebooks](./docs/refs/colab_reference.md)
<!-- TODO: Mover documentação da metodologia CRISP-DM para docs/ -->
<!-- TODO: Mover descrição do dataset para docs/ -->
<!-- TODO: Adicionar documentação sobre treinamento do Reranker -->
<!-- TODO: Adicionar documentação do pipeline RAG -->

Esses documentos servem como guia técnico do projeto durante toda a implementação.

---

## 🏗️ **Visão Geral do Pipeline Proposto**

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

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Datasets](https://img.shields.io/badge/Datasets-Data%20Sources-4ABDAC?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=flat&logo=numpy&logoColor=white)
![Regex](https://img.shields.io/badge/Regex-Text%20Cleaning-critical?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-blueviolet?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-005571?style=flat)
![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-2E8B57?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat&logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00?style=flat&logo=googlecolab&logoColor=black)
![GitHub](https://img.shields.io/badge/GitHub-Version%20Control-181717?style=flat&logo=github&logoColor=white)
