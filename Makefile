# -------- helper targets (type `make help`) --------
help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | sed 's/:.*//' | column

setup:          ## install deps into current interpreter
	pip install -r requirements.txt && pip install -r requirements-dev.txt

preprocess:     ## encode categoricals, train/test split → data/processed/
	python -m src.preprocessing

train:          ## fit KNN, LDA, LR – saves pickles under artifacts/
	python -m src.train_models

ensemble:       ## apply Algorithm 1 & save y_pred.npy
	python -m src.ensemble

evaluate:       ## compute metrics.json + plots
	python -m src.evaluate

clean:          ## wipe derived artefacts
	rm -rf artifacts/* data/processed/* results/*

.PHONY: help setup preprocess train ensemble evaluate clean
