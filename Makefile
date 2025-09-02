.PHONY: build new iterate lint fmt test index

build:
	docker compose build app

new:
	docker compose run --rm app new --prompt "$(PROMPT)" --out /workspace/out/$(NAME)

iterate:
	docker compose run --rm app iterate --prompt "$(PROMPT)" --repo /workspace/out/$(NAME)

lint:
	docker compose run --rm app lint --repo /workspace/out/$(NAME)

fmt:
	docker compose run --rm app fmt --repo /workspace/out/$(NAME)

test:
	docker compose run --rm app test --repo /workspace/out/$(NAME)

index:
	docker compose run --rm app index --repo /workspace/out/$(NAME)
