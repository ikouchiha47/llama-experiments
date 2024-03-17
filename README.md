# RAG-Doll

Using TinyLlama to build a RAG with llama-cpp-python.

## Requirements

- create an account and access token for hugging_face. Install instructions on
  [welcome page](https://huggingface.co/welcome)
- `pip install huggingface_hub`
- `huggingface-cli login`
- using TinyLlama Chat with pre-quantised model from [link](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
- using a `GGUF` model instead of `GPTQ` (to run on CPU)
