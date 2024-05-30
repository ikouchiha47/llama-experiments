setup:
	# turn on metals for Apple GPU
	CMAKE_ARGS="-DLLAMA_METAL_EMBED_LIBRARY=ON -DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir --force-reinstall
	pip install -r requirements.txt


run:
	python3 main.py

freeze:
	pip freeze > requirements.lock.txt

