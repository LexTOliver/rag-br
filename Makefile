# =========================
# Configurações globais
# =========================
PYTHON := uv run python
SRC := src
SCRIPTS := scripts

.DEFAULT_GOAL := help

# =========================
# Targets
# =========================

help:
	@echo "Targets disponíveis:"
	@echo "	 make setup			-> Instala o gerenciador de pacotes e inicializa o projeto"
	@echo "  make install       -> instala dependências"
	@echo "  make lint          -> lint do código"
	@echo "  make clean         -> limpa arquivos temporários"
	@echo "  make ingest        -> executa ingestão de dados com pré-processamento"
	@echo "  make index         -> executa vetorização e indexação de documentos"
# 	@echo "  make chunk         -> chunking de documentos"
# 	@echo "  make embed         -> geração de embeddings"
# 	@echo "  make train         -> treino do reranker"
# 	@echo "  make evaluate      -> avaliação do reranker"
# 	@echo "  make api           -> sobe a API"
# 	@echo "  make test          -> roda testes"


setup:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv init

install:
	uv pip install -e .

ingest:
	$(PYTHON) $(SCRIPTS)/ingest.py

index:
	$(PYTHON) $(SCRIPTS)/build_index.py

# chunk:
# 	$(PYTHON) -m $(SRC).ingest.chunking

# embed:
# 	$(PYTHON) -m $(SRC).ingest.embed

# train:
# 	$(PYTHON) -m $(SRC).training.train_reranker

# evaluate:
# 	$(PYTHON) -m $(SRC).training.evaluate_reranker

# api:
# 	$(PYTHON) -m $(SRC).api.main

# test:
# 	pytest -q

lint:
	uv run ruff check $(SRC) $(SCRIPTS) --fix
	uv run ruff format $(SRC) $(SCRIPTS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
