from llama_index.llms.llama_cpp import LlamaCPP as LlamaCpp
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from huggingface_hub import hf_hub_download
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def suppress_error_output(func):
    def wrapper(*args, **kwargs):
        with suppress_stdout_stderr():
            return func(*args, **kwargs)

    return wrapper


class TinyLlamaModel:
    model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    model_file = "tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf"


class MistralModel:
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    model_file = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"


class LlamaChatModel:
    model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_file = "llama-2-7b-chat.Q5_K_S.gguf"


model = TinyLlamaModel()

tiny_model_name = model.model_name
tiny_model_file = model.model_file


class TinyLlm:
    @suppress_error_output
    def __init__(self, model_name=tiny_model_name, model_file=tiny_model_file):
        self.model_name = model_name
        self.model_file = model_file
        self.verbose = False

        self.model_path = hf_hub_download(model_name, filename=model_file)

        self.model = LlamaCpp(
            model_path=self.model_path,
            temperature=0.1,
            # messages_to_prompt=messages_to_prompt,
            # completion_to_prompt=completion_to_prompt,
            model_kwargs={
                "stop": ["[/INST]", "None", "User:"],
                "context_window": 2048,
                "n_ctx": 1024,
                "n_batch": 100,
                "n_threads": 6,
                "n_gpu_layers": 1,
                # "low_memory": True,
                # repeat_penalty = 1.2,
            },
            verbose=self.verbose,
        )
