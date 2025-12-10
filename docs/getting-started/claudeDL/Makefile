.PHONY: help setup start stop restart clean test test-unit test-integration test-db test-llm logs shell-postgres shell-neo4j

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: ## Initial setup - create .env and start databases
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file - please edit with your configuration"; \
	fi
	@chmod +x init-scripts/*.sh
	@docker-compose up -d
	@echo "Waiting for databases to be ready..."
	@sleep 10
	@docker-compose ps

start: ## Start all database services
	docker-compose up -d
	@echo "Databases started. Check status with: make status"

stop: ## Stop all database services
	docker-compose stop

restart: ## Restart all database services
	docker-compose restart

status: ## Show status of all services
	docker-compose ps

clean: ## Stop and remove all containers and volumes (DESTROYS DATA!)
	@echo "WARNING: This will destroy all database data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
	fi

logs: ## Show logs from all services
	docker-compose logs -f

logs-postgres: ## Show PostgreSQL logs
	docker-compose logs -f postgres

logs-neo4j: ## Show Neo4j logs
	docker-compose logs -f neo4j

shell-postgres: ## Open PostgreSQL shell
	docker exec -it rag_postgres psql -U rag_user -d rag_factory

shell-neo4j: ## Open Neo4j Cypher shell
	docker exec -it rag_neo4j cypher-shell -u neo4j -p rag_password

test: ## Run all tests (respects RUN_DB_TESTS and RUN_LLM_TESTS env vars)
	pytest -v

test-unit: ## Run only unit tests (no external dependencies)
	pytest tests/unit/ -v

test-integration: ## Run integration tests (requires services)
	RUN_DB_TESTS=true pytest tests/integration/ -v

test-db: ## Run database tests only
	RUN_DB_TESTS=true pytest -m requires_db -v

test-llm: ## Run LLM tests only
	RUN_LLM_TESTS=true pytest -m requires_llm -v

test-full: ## Run all tests with all services
	RUN_DB_TESTS=true RUN_LLM_TESTS=true pytest -v

test-coverage: ## Run tests with coverage report
	RUN_DB_TESTS=true pytest --cov=src --cov-report=html --cov-report=term -v

test-fast: ## Run fast tests only (skip slow integration tests)
	pytest -m "not slow" -v

backup-postgres: ## Backup PostgreSQL database
	docker exec rag_postgres pg_dump -U rag_user rag_factory > backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "Backup created: backup_$$(date +%Y%m%d_%H%M%S).sql"

restore-postgres: ## Restore PostgreSQL database (specify FILE=backup.sql)
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make restore-postgres FILE=backup.sql"; \
		exit 1; \
	fi
	docker exec -i rag_postgres psql -U rag_user rag_factory < $(FILE)

health-check: ## Check health of all services
	@echo "Checking PostgreSQL..."
	@docker exec rag_postgres pg_isready -U rag_user || echo "PostgreSQL not ready"
	@echo "Checking Neo4j..."
	@docker exec rag_neo4j cypher-shell -u neo4j -p rag_password "RETURN 1" || echo "Neo4j not ready"
	@echo "Checking Ollama (if configured)..."
	@curl -s $$OLLAMA_BASE_URL/api/tags > /dev/null && echo "Ollama OK" || echo "Ollama not available"

install: ## Install Python dependencies
	pip install -e ".[dev]"

install-full: ## Install Python dependencies with all optional dependencies
	pip install -e ".[dev,torch,onnx]"

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

lint: ## Run linting checks
	black --check src/ tests/
	isort --check src/ tests/
	flake8 src/ tests/
	mypy src/

clean-python: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

monitor: ## Monitor resource usage of databases
	docker stats rag_postgres rag_neo4j
