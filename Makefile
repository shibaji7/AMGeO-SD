# AMGeO-SD - SuperDARN scatter classification (fitacf++)
# Everything is driven from here: run `make` (or `make help`) to see the targets.

CONDA_ENV ?= amgeo
RUN       := conda run --no-capture-output -n $(CONDA_ENV)
PY        := $(RUN) python
PYTEST    := $(RUN) python -m pytest

# Radar-day folders under tests/data (one folder per radar/day; `archives/` is skipped).
DATA ?= $(filter-out %archives/,$(wildcard tests/data/*/))

.DEFAULT_GOAL := help
.PHONY: help env env-update env-remove install data-unpack data-info data-clean test test-unit test-real test-cov clean

##@ Help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<target>\033[0m [VAR=value]\n"} \
	  /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 } \
	  /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	@printf "\nVariables\n  CONDA_ENV=%s   conda environment used for every command\n  DATA=\"%s\"   radar-day folders for data-info\n\n" "$(CONDA_ENV)" "$(DATA)"

##@ Environment
env: ## Create the conda env (Python 3.12) from environment.yml and install the package
	conda env create -n $(CONDA_ENV) -f environment.yml

env-update: ## Update an existing env after environment.yml / pyproject.toml changed
	conda env update -n $(CONDA_ENV) -f environment.yml --prune

env-remove: ## Delete the conda env
	conda env remove -y -n $(CONDA_ENV)

install: ## (Re)install amgeo_sd in editable mode into the env
	$(PY) -m pip install -e . --no-deps

##@ Data
data-unpack: ## Unzip tests/data/archives/*.zip into tests/data/<name>/ (git-ignored)
	@set -e; for z in tests/data/archives/*.zip; do \
	  n=$$(basename $$z _fitacf.zip); mkdir -p tests/data/$$n; unzip -q -o $$z -d tests/data/$$n; echo "unpacked $$z -> tests/data/$$n"; \
	done

data-info: ## Summarise fitacf data (beams, scans, echoes) per radar-day folder; DATA="dir1 dir2"
	@for d in $(DATA); do echo "== $$d"; $(PY) -m amgeo_sd info $$d; done

data-clean: ## Delete unpacked tests/data/<name>/ folders that can be restored from tests/data/archives/ (keeps archives)
	@set -e; for d in $(DATA); do n=$$(basename $$d); \
	  if [ -f tests/data/archives/$${n}_fitacf.zip ]; then rm -rf $$d; echo "removed $$d (restore: make data-unpack)"; \
	  else echo "kept $$d (no tests/data/archives/$${n}_fitacf.zip to restore it from)"; fi; \
	done

##@ Quality
test: ## Run the whole test suite (real-data tests skip if tests/data is empty)
	$(PYTEST) -q

test-unit: ## Fast tests on in-memory records only
	$(PYTEST) -q -m "not realdata"

test-real: ## Tests on the real fitacf files in tests/data (bks mid-lat, gbr high-lat)
	$(PYTEST) -q -m realdata

test-cov: ## Test suite with coverage report
	$(PYTEST) -q --cov=amgeo_sd --cov-report=term-missing

clean: ## Remove caches, coverage and build artefacts (never touches tests/data or _bkp)
	rm -rf .pytest_cache .coverage build dist src/*.egg-info
	find . -name __pycache__ -type d -not -path "./_bkp/*" -prune -exec rm -rf {} +
