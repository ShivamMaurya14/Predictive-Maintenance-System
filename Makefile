
# Makefile for Predictive Maintenance System

PYTHON = python3
PIP = pip
STREAMLIT = streamlit

.PHONY: install run train clean

install:
	$(PIP) install -r requirements.txt

run:
	$(STREAMLIT) run app.py

train:
	$(PYTHON) retrain_model.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f model.joblib
