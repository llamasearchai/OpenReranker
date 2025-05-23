.PHONY: help install install-dev test test-unit test-api test-integration lint format type-check coverage clean build docker-build docker-run setup-env docs security

# Default target
help:
	@echo "OpenReranker Development Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  install          Install package"
	@echo "  install-dev      Install package in development mode with all dependencies"
	@echo "  setup-env        Set up development environment"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests"
	@echo "  test-api         Run API tests"
	@echo "  test-integration Run integration tests"
	@echo "  coverage         Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security         Run security checks"
	@echo ""
	@echo "Development:"
	@echo "  run              Run the development server"
	@echo "  run-prod         Run the production server"
	@echo "  clean            Clean build artifacts"
	@echo "  build            Build package"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  docker-compose   Run with docker-compose"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"

# Installation
install:
	pip install .

install-dev:
	pip install -e ".[dev]"
	pip install pytest pytest-cov pytest-mock pytest-asyncio httpx
	pip install black isort flake8 mypy
	pip install bandit safety
	pip install tox

setup-env: install-dev
	@echo "Development environment set up successfully!"

# Testing
test:
	python run_tests.py --all

test-unit:
	pytest tests/ -v

test-api:
	python run_tests.py --api

test-integration:
	pytest tests/test_integrations.py -v

coverage:
	pytest tests/ --cov=open_reranker --cov-report=html --cov-report=term-missing

# Code quality
lint:
	black --check --diff open_reranker tests
	isort --check-only --diff open_reranker tests
	flake8 open_reranker tests

format:
	black open_reranker tests
	isort open_reranker tests

type-check:
	mypy open_reranker

security:
	bandit -r open_reranker
	safety check

# Development server
run:
	uvicorn open_reranker.main:app --host 0.0.0.0 --port 8000 --reload

run-prod:
	uvicorn open_reranker.main:app --host 0.0.0.0 --port 8000

# Build and packaging
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# Docker
docker-build:
	docker build -t open-reranker .

docker-run:
	docker run -p 8000:8000 open-reranker

docker-compose:
	docker-compose up --build

docker-compose-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Documentation
docs:
	@echo "Building documentation..."
	@echo "Documentation build not implemented yet"

# Tox testing
tox:
	tox

tox-lint:
	tox -e lint

tox-type-check:
	tox -e type-check

tox-coverage:
	tox -e coverage

# Quick development workflow
dev-check: format lint type-check test-unit
	@echo "✅ Development checks passed!"

# CI/CD workflow
ci: lint type-check test coverage security
	@echo "✅ CI checks passed!"

# Release workflow
release: clean build
	@echo "📦 Package built successfully!"
	@echo "To upload to PyPI, run: twine upload dist/*" 