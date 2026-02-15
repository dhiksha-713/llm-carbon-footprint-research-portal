.PHONY: install download ingest query eval-baseline eval-enhanced eval-both report serve ui clean clean-all all help

# Load .env if present (host/port/etc. driven from there)
-include .env
export

# Use venv Python/pip if venv exists, otherwise fall back to system
VENV_DIR := venv
ifneq ($(wildcard $(VENV_DIR)/bin/python),)
  PYTHON := $(VENV_DIR)/bin/python
  PIP    := $(VENV_DIR)/bin/pip
else
  PYTHON := python3
  PIP    := pip3
endif

API_HOST ?= 0.0.0.0
API_PORT ?= 8000
STREAMLIT_PORT ?= 8501

help:
	@echo "Targets: install download ingest query eval-baseline eval-enhanced eval-both report serve ui clean all"

install:
	$(PIP) install -r requirements.txt

download:
	$(PYTHON) -m src.ingest.download_sources

ingest:
	$(PYTHON) -m src.ingest.ingest

query:
	$(PYTHON) -m src.rag.rag --query "What are the major sources of carbon emissions in LLM training?"

eval-baseline:
	$(PYTHON) -m src.eval.evaluation --mode baseline

eval-enhanced:
	$(PYTHON) -m src.eval.evaluation --mode enhanced

eval-both:
	$(PYTHON) -m src.eval.evaluation --mode both

report:
	$(PYTHON) -m src.eval.generate_report

serve:
	$(PYTHON) -m uvicorn src.app.app:app --host $(API_HOST) --port $(API_PORT) --reload

ui:
	$(PYTHON) -m streamlit run src/app/streamlit_ui.py --server.port $(STREAMLIT_PORT)

all: install download ingest eval-both report
	@echo "Done. See report/phase2/evaluation_report.md"

clean:
	rm -rf data/processed/ logs/ outputs/ report/phase2/
	@echo "Cleaned generated artifacts (index, logs, outputs, report)"

clean-all: clean
	rm -rf data/raw/*.pdf
	@echo "Also removed downloaded PDFs - next run will re-download everything"
