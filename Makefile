.DEFAULT_GOAL := help
PY ?= python
PKG := chondrotwin

.PHONY: help install dev lint type test smoke docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## install the package and runtime dependencies
	$(PY) -m pip install -e .

dev: ## install development extras and pre-commit hooks
	$(PY) -m pip install -e ".[dev,backbone,medio,export]"
	pre-commit install

lint: ## run ruff, black --check and isort --check
	ruff check .
	black --check .
	isort --check-only .

type: ## run mypy in strict mode
	$(PY) -m mypy $(PKG)

test: ## run the full proving suite
	$(PY) -m pytest

smoke: ## train two steps on the unit-test recipe
	$(PY) -m $(PKG).gallery.train +experiment=_unittest

docker: ## build the container image
	docker build -t $(PKG):0.1.0 .

clean: ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
