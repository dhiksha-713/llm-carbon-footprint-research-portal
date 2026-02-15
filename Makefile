.PHONY: install download ingest query eval-baseline eval-enhanced eval-both report serve ui clean all help

# Load .env if present (host/port/etc. driven from there)
-include .env
export

API_HOST ?= 0.0.0.0
API_PORT ?= 8000
STREAMLIT_PORT ?= 8501

help:
	@echo "Targets: install download ingest query eval-baseline eval-enhanced eval-both report serve ui clean all"

install:
	pip install -r requirements.txt

download:
	python -m src.ingest.download_sources

ingest:
	python -m src.ingest.ingest

query:
	python -m src.rag.rag --query "What are the major sources of carbon emissions in LLM training?"

eval-baseline:
	python -m src.eval.evaluation --mode baseline

eval-enhanced:
	python -m src.eval.evaluation --mode enhanced

eval-both:
	python -m src.eval.evaluation --mode both

report:
	python -m src.eval.generate_report

serve:
	uvicorn src.app.app:app --host $(API_HOST) --port $(API_PORT) --reload

ui:
	streamlit run src/app/streamlit_ui.py --server.port $(STREAMLIT_PORT)

all: install download ingest eval-both report
	@echo "Done. See report/phase2/evaluation_report.md"

clean:
	rm -rf data/processed/ logs/ outputs/
	@echo "Cleaned generated artifacts"
