# =========================
# Configurações globais
# =========================
PYTHON := python3
SRC := src

.DEFAULT_GOAL := help

# =========================
# Targets
# =========================

help:
	@echo "Targets disponíveis:"
	@echo "  make install       -> instala dependências"
	@echo "  make ingest        -> executa ingestão de dados com pré-processamento"
	@echo "  make lint          -> lint do código"
	@echo "  make clean         -> limpa arquivos temporários"
# 	@echo "  make chunk         -> chunking de documentos"
# 	@echo "  make embed         -> geração de embeddings"
# 	@echo "  make index         -> construção do índice"
# 	@echo "  make train         -> treino do reranker"
# 	@echo "  make evaluate      -> avaliação do reranker"
# 	@echo "  make api           -> sobe a API"
# 	@echo "  make test          -> roda testes"

install:
	pip install -e .

ingest:
	$(PYTHON) -m $(SRC).ingest.load_dataset

# chunk:
# 	$(PYTHON) -m $(SRC).ingest.chunking

# embed:
# 	$(PYTHON) -m $(SRC).ingest.embed

# index:
# 	$(PYTHON) -m $(SRC).ingest.build_index

# train:
# 	$(PYTHON) -m $(SRC).training.train_reranker

# evaluate:
# 	$(PYTHON) -m $(SRC).training.evaluate_reranker

# api:
# 	$(PYTHON) -m $(SRC).api.main

# test:
# 	pytest -q

lint:
	ruff format $(SRC)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
