.PHONY: dev verify smoke docker-up docker-down dataset

dev:
	./scripts/dev.sh

verify:
	./scripts/verify.sh

smoke:
	./scripts/smoke.sh

docker-up:
	mkdir -p data/dataset data/annotations data/longitudinal data/runtime
	LOCAL_UID="$$(id -u)" LOCAL_GID="$$(id -g)" docker compose up --build

docker-down:
	docker compose down

dataset:
	@test -n "$(INPUT)" || (echo "INPUT=/path/to/archive.rar is required" >&2; exit 2)
	@test -n "$(OUTPUT)" || (echo "OUTPUT=/secure/path/dataset is required" >&2; exit 2)
	@test -n "$$OSSEO_PSEUDONYM_KEY" || (echo "OSSEO_PSEUDONYM_KEY is required" >&2; exit 2)
	backend/.venv/bin/osseo-apex "$(INPUT)" --output "$(OUTPUT)" $(if $(PROTOCOL),--protocol "$(PROTOCOL)",)
