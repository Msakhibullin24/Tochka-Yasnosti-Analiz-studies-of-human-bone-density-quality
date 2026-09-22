.PHONY: dev verify smoke clean-generated docker-up docker-down dataset specialists specialists-verify dev-specialists

SPECIALIST_MODEL ?= data/specialists/runs/convnextv2-bce-v1

dev-specialists:
	@test -f "$(SPECIALIST_MODEL)/model.json" -o -f "$(SPECIALIST_MODEL)/suite.json" || (echo "Train a specialist first; see competition/docs/SPECIALISTS.md" >&2; exit 2)
	DXAQC_SPECIALIST_PATH="$(abspath $(SPECIALIST_MODEL))" ./scripts/dev.sh

specialists:
	bash scripts/setup_specialists.sh

specialists-verify:
	backend/.venv/bin/python backend/scripts/fetch_specialists.py --verify

dev:
	./scripts/dev.sh

verify:
	./scripts/verify.sh

smoke:
	./scripts/smoke.sh

clean-generated:
	bash scripts/clean_generated.sh $(if $(APPLY),--apply,)

docker-up:
	mkdir -p data/runtime data/input
	LOCAL_UID="$$(id -u)" LOCAL_GID="$$(id -g)" docker compose up --build

docker-down:
	docker compose down

dataset:
	@test -n "$(INPUT)" || (echo "INPUT=/path/to/archive.rar is required" >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo "OUTPUT=/secure/path/dataset is required" >&2; exit 2)
	@test -n "$$OSSEO_PSEUDONYM_KEY" || (echo "OSSEO_PSEUDONYM_KEY is required" >&2; exit 2)
	backend/.venv/bin/osseo-apex "$(INPUT)" --output "$(OUTPUT)" $(if $(PROTOCOL),--protocol "$(PROTOCOL)",)

.PHONY: train-specialists metrics-specialists
train-specialists:
	@test -n "$(INPUT)" -a -n "$(OUTPUT)" || (echo "INPUT=dataset_directory OUTPUT=new_experiment_directory are required" >&2; exit 2)
	bash scripts/train_specialists.sh "$(INPUT)" "$(OUTPUT)"

METRICS_DIR ?= data/specialists/reports/qc-v2
METRICS_PORT ?= 8092
metrics-specialists:
	@test -f "$(METRICS_DIR)/index.html" || (echo "Generate the metrics report first." >&2; exit 2)
	backend/.venv/bin/python -m http.server "$(METRICS_PORT)" --bind 127.0.0.1 --directory "$(METRICS_DIR)"

.PHONY: advance-specialists
advance-specialists:
	@test -n "$(INPUT)" -a -n "$(OUTPUT)" || (echo "INPUT=dataset_directory OUTPUT=new_experiment_directory are required" >&2; exit 2)
	bash scripts/advance_specialists.sh "$(INPUT)" "$(OUTPUT)"
