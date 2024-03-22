setup:
	CMAKE_ARGS="-DLLAMA_METAL=off" pip install llama-cpp-python
	pip install -r requirements.txt

pg.install:
	LDFLAGS="-L/opt/homebrew/opt/openssl@3/lib" CPPFLAGS="-I/opt/homebrew/opt/openssl@3/include" pip install psycopg2

read:
	TOKENIZERS_PARALLELISM=true python3 main.py read

run.cli:
	python3 main.py run cli

run.web:
	streamlit run main.py run web

freeze:
	pip freeze > requirements.lock.txt

download.models:
	python3 hugfacemodels.py

pg.up:
	docker-compose -f pgvector.docker-compose.yaml up -d

pg.down:
	docker-compose -f pgvector.docker-compose.yaml down
