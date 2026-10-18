.PHONY: install test lint benchmark api
install:
	python -m pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests
benchmark:
	power-price benchmark --rows 26280 --output artifacts
api:
	uvicorn power_price.api:app --host 0.0.0.0 --port 8000
