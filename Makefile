.PHONY: install quick full

install:
	python -m pip install -r requirements.txt

quick:
	python scripts/data.py --limit 10 --skip-query-execution
	python scripts/compute_idf_text2cypher.py --limit 10
	python scripts/train_csencoder.py --limit 10 --epochs 2 --batch-size 4 --seed 42

full:
	python scripts/data.py --skip-query-execution
	python scripts/compute_idf_text2cypher.py
	python scripts/train_csencoder.py --epochs 5 --batch-size 32 --seed 42
