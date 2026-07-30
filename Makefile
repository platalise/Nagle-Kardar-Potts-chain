.PHONY: install install-ml test test-slow lint replot fullscale clean paper help

help:
	@echo "Common targets:"
	@echo "  make install      - pip install -e . (core deps)"
	@echo "  make install-ml   - pip install -e .[ml] (adds torch + torch_geometric)"
	@echo "  make test         - fast pytest"
	@echo "  make test-slow    - full pytest incl. slow MC test"
	@echo "  make replot       - rebuild all figures from data/"
	@echo "  make fullscale    - regenerate ALL data + figures (heavy, needs GPU)"
	@echo "  make paper        - compile paper/manuscript.tex"
	@echo "  make clean        - remove build artifacts and top-level regenerated files"

install:
	pip install -e ".[dev]"

install-ml:
	pip install -e ".[dev,ml]"

test:
	pytest -q -m "not slow"

test-slow:
	pytest -q

replot:
	python scripts/replot_all.py

fullscale:
	bash scripts/run_fullscale.sh

paper:
	cd paper && pdflatex -interaction=nonstopmode manuscript.tex && \
	           pdflatex -interaction=nonstopmode manuscript.tex

clean:
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	find . -type f -name '*.pyc' -delete
	# regenerable outputs at repo root (data/, figures/, paper/ are kept)
	rm -f ./*.csv ./*.npz ./*.json ./*.pdf ./*.png
