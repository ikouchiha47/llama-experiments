huggface:
	pip install huggingface_hub
	huggingface-cli login

install.requirements:
	# pip install jupyterlab; pip install notebook; pip install voila
	pip install huggingface_hub
	pip install transformers
	pip install langchain
	pip install accelerate
	pip install optimum
	pip install peft bitsandbytes trl
	# BUILD_CUDA_EXT=0 pip install auto-gptq
	CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

