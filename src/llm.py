# from langchain_community.llms import LlamaCpp
from llama_index.llms.llama_cpp import LlamaCPP as LlamaCpp
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

from langchain_core.callbacks import StreamingStdOutCallbackHandler

from huggingface_hub import hf_hub_download
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


tiny_model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
tiny_model_file = "tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf"


class TinyLlm:
    def __init__(self, model_name=tiny_model_name, model_file=tiny_model_file):
        self.model_name = model_name
        self.model_file = model_file
        self.model_path = hf_hub_download(model_name, filename=model_file)
        self.verbose = False

        with suppress_stdout_stderr():
            self.model = LlamaCpp(
                model_path=self.model_path,
                temperature=0,
                # messages_to_prompt=messages_to_prompt,
                # completion_to_prompt=completion_to_prompt,
                model_kwargs={
                    "stop": ["[/INST]", "None", "User:"],
                    "context_window": 2048,
                    "n_ctx": 1024,
                    "n_batch": 100,
                    "n_threads": 8,
                    "n_gpu_layers": 0,
                    "callbacks": [StreamingStdOutCallbackHandler()],
                    "low_memory": True,
                    # repeat_penalty = 1.2,
                },
                verbose=self.verbose,
            )
