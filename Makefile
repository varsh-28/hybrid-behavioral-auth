.PHONY: install run test serve
install:
	pip install -r requirements.txt
run:
	python scripts/run_pipeline.py
test:
	python -m pytest -q tests
serve:
	PYTHONPATH=src uvicorn hba.api:app --reload
