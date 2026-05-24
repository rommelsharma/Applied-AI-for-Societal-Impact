.PHONY: setup install dev run stop test test-cov docker-build docker-run docker-compose clean

setup:
	bash scripts/setup.sh

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

run:
	uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

stop:
	@pkill -f "uvicorn backend.app" && echo "Server stopped." || echo "Server was not running."

download-models:
	python scripts/download_models.py

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=backend --cov=configs --cov-report=term-missing

docker-build:
	docker build -f docker/Dockerfile -t tts-agent:latest .

docker-run:
	docker run --gpus all -p 8000:8000 \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/voice_samples:/app/voice_samples \
		-v $(PWD)/outputs:/app/outputs \
		--env-file .env \
		tts-agent:latest

docker-compose:
	docker compose -f docker/docker-compose.yml up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
