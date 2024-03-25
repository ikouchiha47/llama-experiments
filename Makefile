setup:
	CMAKE_ARGS="-DLLAMA_METAL=off" pip install llama-cpp-python
	pip install -r requirements.txt

setup.gpu:
	CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
	pip install tensorflow[and-cuda]
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

