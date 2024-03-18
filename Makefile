huggface:
	pip install huggingface_hub
	huggingface-cli login

install.requirements:
	# pip install jupyterlab; pip install notebook; pip install voila
	pip install transformers huggingface_hub accelerate bitsandbytes peft trl
	pip install langchain optimum
	# BUILD_CUDA_EXT=0 pip install auto-gptq
	CMAKE_ARGS="-DLLAMA_METAL=off" pip install llama-cpp-python

