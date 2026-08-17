.PHONY: help install install-local test smoke index index-offline ask eval eval-full api ui docker-build docker-up clean

PY ?= python3
CORPUS ?= data/corpus
INDEX ?= artifacts/index
EVALSET ?= eval/evalset.jsonl
RESULTS ?= artifacts/results
Q ?= How do I center a div with flexbox?

help:
	@echo "install        install the package (core, numpy only)"
	@echo "install-local  install with local embeddings + reranker (downloads torch)"
	@echo "test           run the unit suite (no network, no API key)"
	@echo "smoke          build an offline index and query it end to end"
	@echo "index          build the index with real embeddings"
	@echo "eval           run the retrieval sweep and write the results table"
	@echo "eval-full      sweep + reranking row + 25 LLM-judged answers"
	@echo "api            serve the FastAPI backend on :8000"
	@echo "ui             serve the Streamlit UI on :8501"
	@echo "clean          delete build artifacts (index + results)"

install:
	$(PY) -m pip install -e ".[api,ui,dev]"

install-local:
	$(PY) -m pip install -e ".[local,api,ui,dev]"

# Runs on a bare interpreter: no pytest, no network, no API key. If this ever
# needs a key to pass, the test is testing the wrong thing.
test:
	$(PY) -m unittest discover -s tests -t . -v

# Proves the whole pipeline works without downloading a model or spending a
# cent: the hash embedder is a deterministic stand-in for a real one.
smoke:
	EMBEDDING_PROVIDER=hash $(PY) scripts/build_index.py --corpus $(CORPUS) --out $(INDEX) --provider hash
	EMBEDDING_PROVIDER=hash $(PY) scripts/ask.py "$(Q)" --index $(INDEX) --retrieval-only
	EMBEDDING_PROVIDER=hash $(PY) scripts/run_eval.py --index $(INDEX) --evalset $(EVALSET) --out $(RESULTS)

index:
	$(PY) scripts/build_index.py --corpus $(CORPUS) --out $(INDEX)

index-offline:
	$(PY) scripts/build_index.py --corpus $(CORPUS) --out $(INDEX) --provider hash

ask:
	$(PY) scripts/ask.py "$(Q)" --index $(INDEX)

eval:
	$(PY) scripts/run_eval.py --index $(INDEX) --evalset $(EVALSET) --out $(RESULTS)

eval-full:
	$(PY) scripts/run_eval.py --index $(INDEX) --evalset $(EVALSET) --out $(RESULTS) --rerank --faithfulness 25

api:
	$(PY) -m uvicorn app.api:app --reload --port $${PORT:-8000}

ui:
	$(PY) -m streamlit run app/streamlit_app.py

docker-build:
	docker build -t docsrag:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf $(INDEX) $(RESULTS)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
