PYTHON ?= python
.PHONY: install lint test audit benchmark serve docker
install:
	$(PYTHON) -m pip install -e ".[dev,research,export,tracking,diagnostics]"
lint:
	$(PYTHON) -m ruff check chronogrid tests scripts
	$(PYTHON) -m ruff format --check chronogrid tests scripts
	$(PYTHON) -m mypy chronogrid
test:
	$(PYTHON) -m pytest --cov=chronogrid
audit:
	$(PYTHON) -m pip_audit
	$(PYTHON) scripts/audit_gate.py
benchmark:
	$(PYTHON) scripts/benchmark_corrected.py
serve:
	$(PYTHON) -m uvicorn chronogrid.serving.app:app --host 127.0.0.1 --port 8000
docker:
	docker build -f deployment/Dockerfile -t chronogrid:0.2.0 .
