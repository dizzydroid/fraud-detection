# -------- helper targets (type `make help`) --------
help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | sed 's/:.*//'

setup:  ## install deps into current interpreter
	pip install -r requirements.txt && pip install -r requirements-dev.txt

train:  ## fit KNN, LDA, LR – saves pickles under artifacts/
	python -m src.train_models

ensemble:  ## apply Algorithm 1 & save y_pred.npy
	python -m src.ensemble

evaluate: ensemble  ## compute metrics.json + plots
	python -m src.evaluate

clean:
	rm -rf artifacts/* data/processed/* results/*

.PHONY: help setup train ensemble evaluate clean
