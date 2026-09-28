install:
	python -m pip install -r requirements.txt

mysql:
	docker compose up -d mysql

ingest:
	python -m scripts.ingest

run:
	uvicorn app.main:app --reload

test:
	pytest -q
