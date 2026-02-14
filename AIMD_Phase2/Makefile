# Phase 2 RAG — Makefile
# Usage: make <target>
# Requires: ANTHROPIC_API_KEY set in environment

.PHONY: install download ingest query eval-baseline eval-enhanced eval-both report clean all

## Install all dependencies
install:
	pip install -r requirements.txt

## Download PDFs from arXiv and other sources
download:
	python download_sources.py

## Parse, chunk, embed, and index all sources
ingest:
	python ingest.py

## Run a single demo query (baseline RAG)
query:
	python rag.py --query "What are the major sources of carbon emissions in LLM training?"

## Run the full 20-query evaluation — baseline only
eval-baseline:
	python evaluation.py --mode baseline

## Run the full 20-query evaluation — enhanced only  
eval-enhanced:
	python evaluation.py --mode enhanced

## Run full evaluation — both modes (takes ~10-15 min)
eval-both:
	python evaluation.py --mode both

## Generate evaluation report from results
report:
	python generate_report.py

## Full pipeline (after setting ANTHROPIC_API_KEY)
all: install download ingest eval-both report
	@echo "✓ Phase 2 complete. See outputs/evaluation_report.md"

## Remove generated files (keeps data/raw PDFs)
clean:
	rm -rf data/processed/
	rm -rf logs/
	rm -rf outputs/
	@echo "Cleaned processed data, logs, and outputs"
