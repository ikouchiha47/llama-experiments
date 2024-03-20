setup:
	CMAKE_ARGS="-DLLAMA_METAL=off" pip install llama-cpp-python
	pip install -r requirements.txt

read:
	rm -rf datastore/ipl_db
	python3 main.py read

run:
	python3 main.py run

freeze:
	pip freeze > requirements.lock.txt

download.models:
	python3 hugfacemodels.py
