# Variables pour les commandes fréquemment utilisées
PYTHON = python3


all: main

main:
	$(PYTHON) src/main.py

test:
	pytest -s tests/test_value_iteration.py tests/test_analysis.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

.PHONY: all main test clean

