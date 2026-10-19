# AMGeO-SD - SuperDARN scatter classification (fitacf++)
# Everything is driven from here: run `make` (or `make help`) to see the targets.

CONDA_ENV ?= amgeo
RUN       := conda run --no-capture-output -n $(CONDA_ENV)
PY        := $(RUN) python
PYTEST    := $(RUN) python -m pytest

# Radar-day folders under tests/data (one folder per radar/day; `archives/` is skipped).
DATA ?= $(filter-out %archives/,$(wildcard tests/data/*/))

# Pipeline parameters (override on the command line, e.g. `make median RAD=gbr START=... END=...`).
CONFIG ?=
RAD    ?=
START  ?=
END    ?=
INPUT  ?=
OUT    ?=
SID    ?=
JOBS   ?=
THRESH ?=
PTH    ?=
LTH    ?=
UTH    ?=
EXTRA  ?=
COMPARE ?=
GEOMETRY ?=
GMM    ?=
DAY    ?= 2025-02-08
WORK   ?= data/analysis
BEAM   ?=
TIME   ?=
PARAMS ?=
COORDS ?=
FIGDIR ?=

.DEFAULT_GOAL := help
.PHONY: help env env-update env-remove install data-unpack data-info data-clean median median-demo dbscan dbscan-demo out-clean plot-rti plot-fov plot-demo analyse-day config-show lint format test test-unit test-real test-cov clean

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

##@ Pipelines
median: ## fitacf -> fitacf++ (.nc + .fitacf.pl.bz2). CONFIG=file.toml and/or RAD START END [INPUT OUT SID JOBS THRESH PTH LTH UTH EXTRA]
	@test -n "$(CONFIG)" -o \( -n "$(RAD)" -a -n "$(START)" -a -n "$(END)" \) || { echo "usage: make median CONFIG=configs/median_bks_demo.toml   or   make median RAD=bks START=2025-02-08T03:00 END=2025-02-08T04:00 INPUT=tests/data/bks_20250208"; exit 2; }
	$(PY) -m amgeo_sd median $(if $(CONFIG),-c $(CONFIG)) $(if $(RAD),-r $(RAD)) $(if $(START),-s $(START)) $(if $(END),-e $(END)) $(if $(INPUT),-i $(INPUT)) $(if $(OUT),-o $(OUT)) $(if $(SID),-sid $(SID)) $(if $(JOBS),-j $(JOBS)) $(if $(THRESH),-th $(THRESH)) $(if $(PTH),-pth $(PTH)) $(if $(LTH),-lth $(LTH)) $(if $(UTH),-uth $(UTH)) $(if $(GEOMETRY),--geometry $(GEOMETRY)) $(EXTRA)

dbscan: ## DBSCAN/GMM model: label raw echoes with clusters + Ribiero flag -> fitacf++ (.nc + .fitacf.pl.bz2). CONFIG and/or RAD START END [INPUT OUT SID GMM GEOMETRY EXTRA]
	@test -n "$(CONFIG)" -o \( -n "$(RAD)" -a -n "$(START)" -a -n "$(END)" \) || { echo "usage: make dbscan CONFIG=configs/dbscan_bks_demo.toml   or   make dbscan RAD=bks START=2025-02-08T03:00 END=2025-02-08T04:00 INPUT=tests/data/bks_20250208 [GMM=A]"; exit 2; }
	$(PY) -m amgeo_sd dbscan $(if $(CONFIG),-c $(CONFIG)) $(if $(RAD),-r $(RAD)) $(if $(START),-s $(START)) $(if $(END),-e $(END)) $(if $(INPUT),-i $(INPUT)) $(if $(OUT),-o $(OUT)) $(if $(SID),-sid $(SID)) $(if $(GMM),-gmm $(GMM)) $(if $(GEOMETRY),--geometry $(GEOMETRY)) $(EXTRA)

dbscan-demo: ## Run `dbscan` from configs/dbscan_{bks,gbr}_demo.toml (1 h each, tests/data) into data/outputs/demo-dbscan/
	$(MAKE) --no-print-directory dbscan CONFIG=configs/dbscan_bks_demo.toml
	$(MAKE) --no-print-directory dbscan CONFIG=configs/dbscan_gbr_demo.toml

median-demo: ## Run `median` from configs/median_{bks,gbr}_demo.toml (1 h each, tests/data) into data/outputs/demo/
	$(MAKE) --no-print-directory median CONFIG=configs/median_bks_demo.toml
	$(MAKE) --no-print-directory median CONFIG=configs/median_gbr_demo.toml

config-show: ## Print the resolved median parameters (CONFIG=... plus overrides) without running
	$(PY) -m amgeo_sd median --print-config $(if $(CONFIG),-c $(CONFIG)) $(if $(RAD),-r $(RAD)) $(if $(START),-s $(START)) $(if $(END),-e $(END)) $(EXTRA)

out-clean: ## Delete pipeline results in OUT (default data/outputs)
	@o="$(if $(OUT),$(OUT),data/outputs)"; case "$$o" in /|.|..) echo "refusing to remove '$$o'"; exit 2;; esac; rm -rf "$$o"

PLOT_FLAGS = $(if $(GEOMETRY),--geometry $(GEOMETRY)) $(if $(CONFIG),-c $(CONFIG)) $(if $(INPUT),-i $(INPUT)) $(if $(RAD),-r $(RAD)) $(if $(START),-s $(START)) $(if $(END),-e $(END)) $(if $(PARAMS),-p $(PARAMS)) $(if $(FIGDIR),-o $(FIGDIR)) $(EXTRA)

plot-rti: ## RTI of one beam from fitacf or fitacf++ files. CONFIG=file.toml and/or INPUT BEAM [COMPARE START END PARAMS FIGDIR EXTRA]
	@test -n "$(CONFIG)" -o -n "$(INPUT)" || { echo "usage: make plot-rti INPUT=data/outputs/demo/bks/x.fitacf++.nc BEAM=12 [COMPARE=raw_dir PARAMS='v gflg' START=... END=...]   or   CONFIG=configs/plot_rti_demo.toml"; exit 2; }
	$(PY) -m amgeo_sd plot rti $(PLOT_FLAGS) $(if $(BEAM),-b $(BEAM)) $(if $(COMPARE),--compare $(COMPARE))

plot-fov: ## Field-of-view map of one scan from fitacf or fitacf++ files. CONFIG=file.toml and/or INPUT TIME [PARAMS COORDS RAD FIGDIR EXTRA]
	@test -n "$(CONFIG)" -o -n "$(INPUT)" || { echo "usage: make plot-fov INPUT=tests/data/bks_20250208 TIME=2025-02-08T03:30 [PARAMS='v gflg_conv' COORDS=aacgmv2]   or   CONFIG=configs/plot_fov_demo.toml"; exit 2; }
	$(PY) -m amgeo_sd plot fov $(PLOT_FLAGS) $(if $(TIME),-t $(TIME)) $(if $(COORDS),--coords $(COORDS))

plot-demo: median-demo ## Run the demo filter, then draw RTI + FoV figures (raw vs fitacf++) for bks and gbr into data/figures/demo/
	$(MAKE) --no-print-directory plot-rti CONFIG=configs/plot_rti_demo.toml
	$(MAKE) --no-print-directory plot-fov CONFIG=configs/plot_fov_demo.toml
	$(MAKE) --no-print-directory plot-rti INPUT=tests/data/gbr_20250208 COMPARE=data/outputs/demo/gbr/gbr_202502080300_202502080400.fitacf++.nc BEAM=7 START=2025-02-08T03:00 END=2025-02-08T04:00 PARAMS=v FIGDIR=data/figures/demo EXTRA="--name rti_gbr_b07_raw_vs_fitacfpp"
	$(MAKE) --no-print-directory plot-fov INPUT=tests/data/gbr_20250208 TIME=2025-02-08T03:05 PARAMS="v gflg" START=2025-02-08T03:00 END=2025-02-08T03:10 FIGDIR=data/figures/demo EXTRA="--name fov_gbr_0305_raw"

ANALYSIS_DIR = examples/median_day_analysis
ANALYSIS_DATA = $(or $(INPUT),tests/data/$(RAD)_$(subst -,,$(DAY)))
ANALYSIS_FLAGS = --rad $(RAD) --work $(WORK)/$(RAD) --day $(DAY) $(if $(INPUT),--data $(INPUT))

analyse-day: ## Whole-day comparison of the median filter and the DBSCAN model for RAD (DAY, INPUT, WORK): runs both, writes tables + figures to data/analysis/<rad>/
	@test -n "$(RAD)" || { echo "usage: make analyse-day RAD=bks [DAY=2025-02-08 INPUT=tests/data/bks_20250208 WORK=data/analysis]"; exit 2; }
	$(MAKE) --no-print-directory median RAD=$(RAD) START=$(DAY)T00:00 END=$(DAY)T23:59:59 INPUT=$(ANALYSIS_DATA) SID=day OUT=$(WORK)/$(RAD) JOBS=4 EXTRA="--no-fitacf"
	$(MAKE) --no-print-directory dbscan RAD=$(RAD) START=$(DAY)T00:00 END=$(DAY)T23:59:59 INPUT=$(ANALYSIS_DATA) SID=day-dbscan OUT=$(WORK)/$(RAD) EXTRA="--no-fitacf"
	$(PY) $(ANALYSIS_DIR)/analysis_day.py $(ANALYSIS_FLAGS) | tee $(WORK)/$(RAD)/retention.txt
	$(PY) $(ANALYSIS_DIR)/analysis_more.py $(ANALYSIS_FLAGS) | tee $(WORK)/$(RAD)/edges_and_flags.txt
	$(PY) $(ANALYSIS_DIR)/analysis_flags.py $(ANALYSIS_FLAGS) | tee $(WORK)/$(RAD)/flag_physics.txt
	$(PY) $(ANALYSIS_DIR)/plot_day.py $(ANALYSIS_FLAGS) $(if $(BEAM),--beam $(BEAM))

##@ Quality
lint: ## Static checks (ruff): errors, unused code, import order, missing docstrings
	$(RUN) ruff check src tests examples

format: ## Format the code with ruff (the ported legacy geometry modules are excluded)
	$(RUN) ruff format src tests examples
	$(RUN) ruff check --fix src tests examples

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
