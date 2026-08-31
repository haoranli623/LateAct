TECTONIC ?= tectonic
PYTHON ?= python3

.PHONY: all figures clean

all:
	mkdir -p build
	$(TECTONIC) --outdir build --keep-logs main.tex

figures:
	test -n "$(LATEACT_ARTIFACT_ROOT)"
	$(PYTHON) make_figures.py

clean:
	rm -rf build
