.PHONY: all
all: clean cov build

.PHONY: clean
clean:
	rm -rf dist/

.PHONY: test
test:
	poetry run pytest

.PHONY: cov
cov:
	poetry run pytest --cov --cov-report=term-missing

.PHONY: build
build:
	poetry build
