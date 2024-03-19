setup:
	CMAKE_ARGS="-DLLAMA_METAL=off" pip install llama-cpp-python
	pip install -r requirements.txt

run:
	python3 main.py
