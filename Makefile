PYTHON ?= python

.PHONY: setup lint format test train_central train_fed toy_data

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]

format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

train_central:
	$(PYTHON) -m src.train_central

train_fed:
	$(PYTHON) -m src.train_federated

toy_data:
	$(PYTHON) -m src.infer.infer_study --help
