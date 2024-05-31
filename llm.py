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
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def suppress_error_output(func):
    def wrapper(*args, **kwargs):
        with suppress_stdout_stderr():
            return func(*args, **kwargs)

    return wrapper


class ClosedAI:
    modelname = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    filename = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    # modelname = "microsoft/Phi-3-mini-4k-instruct-gguf"
    # filename = "Phi-3-mini-4k-instruct-q4.gguf"

    @suppress_error_output
    def __init__(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.model_path: str = hf_hub_download(self.modelname, filename=self.filename)

        self.model = LlamaCpp(
            model_path=self.model_path,  # noqa: e501
            temperature=0.1,
            repeat_penalty=1.1,
            n_gpu_layers=-1,
            max_tokens=768,
            n_ctx=4000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=False,
            # model_kwargs={"stream": True},
            # stop=["<|endoftext|>"],
            # Verbose is required to pass to the callback manager
        )
