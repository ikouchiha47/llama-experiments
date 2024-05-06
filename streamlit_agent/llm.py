from os import devnull
import os
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import (
    CallbackManager,
    StreamingStdOutCallbackHandler,
)  # noqa: e501

from huggingface_hub import hf_hub_download
from contextlib import contextmanager, redirect_stderr, redirect_stdout

# model_path = "TheBloke/WizardCoder-Python-7B-V1.0-GGUF"
# filename = "wizardcoder-python-7b-v1.0.Q4_K_M.gguf"


@contextmanager
def suppress_stdout_stderr():
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def suppress_error_output(func):
    def wrapper(*args, **kwargs):
        with suppress_stdout_stderr():
            return func(*args, **kwargs)

    return wrapper


# model = "TheBloke/CodeLlama-7B-Python-GGUF"
# filename = "codellama-7b-python.Q4_K_M.gguf"
# filename = "codellama-7b-python.Q3_K_S.gguf"
# model = "TheBloke/CodeLlama-7B-Instruct-GGUF"
# filename = "codellama-7b-instruct.Q4_K_M.gguf"

# model = "Qwen/CodeQwen1.5-7B-Chat-GGUF"
# filename = "codeqwen-1_5-7b-chat-q4_k_m.gguf"


class ClosedAI:
    model = "second-state/StarCoder2-7B-GGUF"
    filename = "starcoder2-7b-Q4_K_M.gguf"

    @suppress_error_output
    def __init__(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.model_path: str = hf_hub_download(self.model, filename=self.filename)

        self.llm = LlamaCpp(
            model_path=self.model_path,  # noqa: e501
            temperature=0.3,
            n_gpu_layers=4,
            max_tokens=768,
            n_ctx=2048,
            top_p=1,
            callback_manager=callback_manager,
            verbose=False,
            stop=["Human:"],
            # Verbose is required to pass to the callback manager
        )
