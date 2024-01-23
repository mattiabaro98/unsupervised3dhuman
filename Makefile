OK_MSG = "Done! ✔"
SHELL = bash

RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(RUN_ARGS):;@:)
ifeq ($(strip $(RUN_ARGS)),)
    RUN_ARGS := .
endif

env: lint_requirements.txt
	@( \
		echo -e "==> Creating virtualenv..."; \
		python3 -m venv lint_env; \
		source ./lint_env/bin/activate; \
		pip install -r lint_requirements.txt \
	)
	@echo $(OK_MSG)

env-clean:
	@rm -rf lint_env/
	@echo $(OK_MSG)

code-format:
	@( \
		source ./lint_env/bin/activate; \
		python3 -m black $(RUN_ARGS); \
	)
	@echo $(OK_MSG)
	@echo $(RUN_ARGS)

code-format-check:
	@( \
		source ./lint_env/bin/activate; \
		python3 -m black $(RUN_ARGS) --check; \
	)
	@echo $(OK_MSG)


flake-lint:
	@( \
		source ./lint_env/bin/activate; \
		python3 -m flake8 $(RUN_ARGS); \
	)
	@echo $(OK_MSG)

pylint:
	@( \
		source ./lint_env/bin/activate; \
		find $(RUN_ARGS) -type f -iname "*.py" -not -path "*/lint_env/*" -not -path "*/env/*" -not -path "*/venv/*" -exec python3 -m pylint {} \; ;\
	)
	@echo $(OK_MSG)

isort:
	@( \
		source ./lint_env/bin/activate; \
		python3 -m isort $(RUN_ARGS); \
	)
	@echo $(OK_MSG)

isort-check:
	@( \
		source ./lint_env/bin/activate; \
		python3 -m isort $(RUN_ARGS) --check-only; \
	)
	@echo $(OK_MSG)

lint:
	@( \
		source ./lint_env/bin/activate; \
		python3 -m isort $(RUN_ARGS); \
		python3 -m black $(RUN_ARGS); \
		python3 -m flake8 $(RUN_ARGS); \
	)
	@echo $(OK_MSG)

clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@echo $(OK_MSG)
