.DEFAULT_GOAL := help

SRC_DIRS := src tests examples
PKG_DIR := src/whytrend

# Package manager: auto | uv | poetry | pip
#   make check              # auto-detect
#   make check TOOL=poetry
#   make install TOOL=pip
TOOL ?= auto

HAS_UV := $(shell command -v uv 2>/dev/null)
HAS_POETRY := $(shell command -v poetry 2>/dev/null)
HAS_VENV := $(wildcard .venv/bin/python)

ifeq ($(TOOL),auto)
  ifneq ($(HAS_UV),)
	TOOL := uv
  else ifneq ($(HAS_POETRY),)
	TOOL := poetry
  else
	TOOL := pip
  endif
endif

ifeq ($(TOOL),uv)
	RUFF := uv run ruff
	MYPY := uv run mypy
	PYTEST := uv run pytest
else ifeq ($(TOOL),poetry)
	RUFF := poetry run ruff
	MYPY := poetry run mypy
	PYTEST := poetry run pytest
else ifeq ($(TOOL),pip)
  ifneq ($(HAS_VENV),)
	RUFF := .venv/bin/ruff
	MYPY := .venv/bin/mypy
	PYTEST := .venv/bin/pytest
  else
	RUFF := ruff
	MYPY := mypy
	PYTEST := pytest
  endif
else
  $(error Unknown TOOL='$(TOOL)'. Use: auto, uv, poetry, or pip)
endif

.PHONY: help install lint format format-check typecheck test check clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  TOOL=auto|uv|poetry|pip   (current: $(TOOL))"
	@echo "  Examples: make check TOOL=poetry"
	@echo "            make install TOOL=pip"

install: ## Install package with dev dependencies
ifeq ($(TOOL),uv)
	uv sync --extra dev
else ifeq ($(TOOL),poetry)
	poetry install --extras dev
else
	pip install -e ".[dev]"
endif

lint: ## Run ruff linter
	$(RUFF) check $(SRC_DIRS)

format: ## Format code with ruff
	$(RUFF) format $(SRC_DIRS)
	$(RUFF) check --fix $(SRC_DIRS)

format-check: ## Check formatting without writing changes
	$(RUFF) format --check $(SRC_DIRS)

typecheck: ## Run mypy
	$(MYPY) $(PKG_DIR)

test: ## Run pytest
	$(PYTEST)

check: lint format-check typecheck test ## Run full local CI suite

clean: ## Remove caches and build artifacts
	rm -rf .mypy_cache .pytest_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
