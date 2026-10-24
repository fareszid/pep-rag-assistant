.PHONY: install ingest index eval app test lint format

install:  ## Install the package, the web app and dev tools
	pip install -e ".[app,dev]"

ingest:  ## Download PEP sources and split them into chunks (~10 s)
	pepbot ingest

index:  ## Embed all chunks on CPU, plus the ablation variant used by `make eval`
	pepbot index --ablation

eval:  ## Retrieval benchmark (offline); add --generation to also grade Claude's answers
	pepbot eval

app:
	streamlit run app/streamlit_app.py

test:
	pytest

lint:
	ruff check src tests app
	ruff format --check src tests app

format:
	ruff check --fix src tests app
	ruff format src tests app
