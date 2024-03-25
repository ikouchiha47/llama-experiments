setup:
	CMAKE_ARGS="-DLLAMA_METAL_EMBED_LIBRARY=ON -DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir --force-reinstall
	pip install -r requirements.txt

setup.gpu:
	pip install llama-cpp-python
	pip install --upgrade -r requirements.txt

read:
	TOKENIZERS_PARALLELISM=true python3 main.py read

run.cli:
	python3 main.py -f ./datasets/each_match_records.csv run -cli

run.web:
	streamlit run main.py run web

freeze:
	pip freeze > requirements.lock.txt

download.models:
	python3 hugfacemodels.py

