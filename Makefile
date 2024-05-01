setup:
	CMAKE_ARGS="-DLLAMA_METAL_EMBED_LIBRARY=ON -DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir --force-reinstall
	pip install -r requirements.txt

koderai:
	streamlit run main.py

dochub:
	streamlit run docuhub.py

freeze:
	pip freeze > requirements.lock.txt

download.models:
	python3 hugfacemodels.py

