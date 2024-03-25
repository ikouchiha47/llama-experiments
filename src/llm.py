import os
from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)


from huggingface_hub import hf_hub_download
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     pipeline,
# )
# from langchain_community.llms import HuggingFacePipeline

# from .st_utils import get_device


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as f_null:
        with redirect_stderr(f_null) as err, redirect_stdout(f_null) as out:
            yield (err, out)


def suppress_error_output(func):
    def wrapper(*args, **kwargs):
        with suppress_stdout_stderr():
            return func(*args, **kwargs)

    return wrapper


class TinyLlamaModelConfig:
    model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    model_file = "tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf"


class MistralModelConfig:
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    model_file = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"


class LlamaChatModelConfig:
    model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_file = "llama-2-7b-chat.Q5_K_S.gguf"


class CodeLlamaModelConfig:
    model_name = "TheBloke/CodeLlama-7B-Instruct-GGUF"
    model_file = "codellama-7b-instruct.Q5_K_M.gguf"
    # model_file = "codellama-7b-instruct.Q4_K_M.gguf"
    # model_name = "TheBloke/CodeLlama-13B-Instruct-GGUF"
    # model_file = "codellama-13b-instruct.Q5_K_S.gguf"


class StarcoderModelConfig:
    model_name = "second-state/StarCoder2-7B-GGUF"
    model_file = "starcoder2-7b-Q5_K_M.gguf"
    checkpoint = "bigcode/starcoder2-7b"


class TinyLlm:
    cfg = CodeLlamaModelConfig()

    # @suppress_error_output
    def __init__(self, model_name=None, model_file=None):
        model = self.cfg

        self.model_name = model.model_name if model_name is None else model_name
        self.model_file = model.model_file if model_file is None else model_file
        self.verbose = False

        print("getting model", self.model_name)
        self.model_path = hf_hub_download(
            self.model_name,
            filename=self.model_file,
        )

        callback_manager = [FinalStreamingStdOutCallbackHandler()]

        self.model = LlamaCpp(
            model_path=self.model_path,
            temperature=0,
            callbacks=callback_manager,
            n_batch=20,
            n_gpu_layers=0,
            streaming=True,
            max_tokens=2048,
            n_ctx=2048,
            # context_window=4096,
            verbose=self.verbose,
        )


class TinyLlmGPU:
    cfg = CodeLlamaModelConfig()

    # @suppress_error_output
    def __init__(self, model_name=None, model_file=None):
        model = self.cfg

        self.model_name = (
            model.model_name if model_name is None else model_name
        )  # noqa: e501
        self.model_file = (
            model.model_file if model_file is None else model_file
        )  # noqa: e501
        self.verbose = os.environ.get("DEBUG") == "true"

        print("getting model", self.model_name)
        self.model_path = hf_hub_download(
            self.model_name,
            filename=self.model_file,
        )

        callback_manager = [FinalStreamingStdOutCallbackHandler()]

        self.model = LlamaCpp(
            model_path=self.model_path,
            temperature=0,
            callbacks=callback_manager,
            n_batch=50,
            n_gpu_layers=5,
            n_ctx=4096,
            streaming=True,
            max_tokens=2048,
            model_kwargs={
                "context_window": 8192,
                "n_gqa": 8,
            },
            verbose=self.verbose,
        )
