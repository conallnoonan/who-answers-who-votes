PY      := PYTHONPATH=src python
DRAWS   ?= 1000
CHAINS  ?= 4
DEMO    := WAWV_DATA_RAW=data/synthetic WAWV_DB=data/demo.duckdb

.PHONY: help install all load overreport turnout estimate evaluate figures report \
        demo dashboard dashboard-demo notebook test lint clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:  ## install the package with all extras
	pip install -e ".[dashboard,notebooks,dev]"

# ---- real data -------------------------------------------------------------
all: load overreport turnout estimate evaluate figures report  ## full pipeline on data/raw/

load:        ## ingest raw files + build SQL tables
	$(PY) -m wawv.load
overreport:  ## RQ1–2
	$(PY) -m wawv.overreport
turnout:     ## RQ3
	$(PY) -m wawv.turnout_ml
estimate:    ## RQ4 estimators (DRAWS=, CHAINS=)
	$(PY) -m wawv.estimators --draws $(DRAWS) --chains $(CHAINS)
evaluate:    ## RQ4 accuracy + paired tests
	$(PY) -m wawv.evaluate
figures:     ## all report figures
	$(PY) -m wawv.figures
report:      ## REPORT.md + README key results
	$(PY) -m wawv.report

# ---- synthetic end-to-end run (no downloads) --------------------------------
demo:  ## full pipeline on synthetic data → reports/demo/
	$(PY) -m wawv.synth
	$(DEMO) $(MAKE) --no-print-directory all DRAWS=500 CHAINS=2

# ---- extras -----------------------------------------------------------------
dashboard:       ## interactive explorer (real results)
	streamlit run app/dashboard.py
dashboard-demo:  ## interactive explorer (demo results)
	$(DEMO) streamlit run app/dashboard.py
notebook:        ## regenerate the MRP diagnostics notebook
	python notebooks/make_notebook.py

test:  ## unit tests
	$(PY) -m pytest -q
lint:  ## ruff
	ruff check src app tests

clean:  ## remove generated outputs (keeps raw data)
	rm -rf data/*.duckdb* data/synthetic models/*.joblib models/*.nc models/demo reports/demo
	rm -f reports/tables/*.csv reports/figures/*.png reports/REPORT.md
