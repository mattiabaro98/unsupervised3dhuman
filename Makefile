OK_MSG = "Done! ✔"
SHELL = bash

env: lint_requirements.txt
	@( \
		echo -e "==> Creating virtualenv..."; \
		python3 -m venv env; \
		source ./env/bin/activate; \
		pip install -r lint_requirements.txt \
	)
	@echo $(OK_MSG)

env-clean:
	@rm -rf env/
	@echo $(OK_MSG)

code-format:
	@( \
		source ./env/bin/activate; \
		python3 -m black .; \
	)
	@echo $(OK_MSG)

code-format-check:
	@( \
		source ./env/bin/activate; \
		python3 -m black . --check; \
	)
	@echo $(OK_MSG)


flake-lint:
	@( \
		source ./env/bin/activate; \
		python3 -m flake8 .; \
	)
	@echo $(OK_MSG)

isort:
	@( \
		source ./env/bin/activate; \
		python3 -m isort .; \
	)
	@echo $(OK_MSG)

isort-check:
	@( \
		source ./env/bin/activate; \
		python3 -m isort . --check-only; \
	)
	@echo $(OK_MSG)

lint:
	@( \
		source ./env/bin/activate; \
		python3 -m isort .; \
		python3 -m black .; \
		python3 -m flake8 .; \
	)
	@echo $(OK_MSG)

clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@echo $(OK_MSG)
