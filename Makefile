COMPOSE_BASE = docker compose -f docker/docker-compose.yml
COMPOSE_DEV = docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml
PYTHON_ENV = conda run --no-capture-output -n crystallmv2_venv
APPTAINER_SIF ?= crystallm-api.sif
APPTAINER_DOCKER_IMAGE ?= crystallm-api
APPTAINER_DOCKER_SOURCE_TAG ?= local
APPTAINER_DOCKER_TAG ?= latest

.PHONY: api-build api-up api-up-build api-up-dev api-up-dev-build api-down api-logs api-health api-test api-test-integration api-apptainer-build api-apptainer-run api-port-check api-apptainer-cif-check

api-build:
	$(COMPOSE_BASE) build api

api-up-build:
	$(COMPOSE_BASE) up -d --build

api-up:
	$(COMPOSE_BASE) up -d

api-up-dev-build:
	$(COMPOSE_DEV) up -d --build

api-up-dev:
	$(COMPOSE_DEV) up -d

api-down:
	$(COMPOSE_BASE) down

api-logs:
	$(COMPOSE_BASE) logs -f api

api-health:
	curl -fsS http://localhost:8000/healthz

api-test:
	$(PYTHON_ENV) python -m tests.api.suite --docker_url http://localhost:8000

api-test-with-integration:
	$(PYTHON_ENV) python -m tests.api.suite --docker_url http://localhost:8000 --integration

api-apptainer-build:
	docker tag $(APPTAINER_DOCKER_IMAGE):$(APPTAINER_DOCKER_SOURCE_TAG) $(APPTAINER_DOCKER_IMAGE):$(APPTAINER_DOCKER_TAG)
	apptainer build $(APPTAINER_SIF) docker-daemon://$(APPTAINER_DOCKER_IMAGE):$(APPTAINER_DOCKER_TAG)

api-apptainer-run: api-port-check
	@if ss -ltn 2>/dev/null | awk 'NR>1 {print $$4}' | grep -Eq '(^|:)8000$$'; then \
	  echo "Port 8000 is already in use. Stop Docker API first with: make api-down or Ctrl+C on apptainer image"; \
	  exit 1; \
	fi
	mkdir -p data outputs
	apptainer run --nv --cleanenv \
	  --bind "$(PWD)/data:/app/data" \
	  --bind "$(PWD)/outputs:/app/outputs" \
	  --bind "$(PWD)/tests/fixtures:/app/tests/fixtures" \
	  --env HF_KEY="$$HF_KEY" \
	  --env WANDB_KEY="$$WANDB_KEY" \
	  $(APPTAINER_SIF)