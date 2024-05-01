from os import devnull
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import (
    CallbackManager,
    StreamingStdOutCallbackHandler,
)  # noqa: e501

from langchain.callbacks.base import BaseCallbackHandler

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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class KoderModel:
    model = "TheBloke/CodeLlama-7B-Instruct-GGUF"
    filename = "codellama-7b-instruct.Q4_K_M.gguf"


class ChatModel:
    model = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"


class ClosedAI:
    # @suppress_error_output
    def __init__(self, model_conf, stop=None):
        if stop is None:
            stop = ["\n\n"]

        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # print(model_conf.model, "Odel....")

        self.model_path = hf_hub_download(
            model_conf.model, filename=model_conf.filename
        )

        self.llm = LlamaCpp(
            model_path=self.model_path,  # noqa: e501
            temperature=0.3,
            n_gpu_layers=4,
            max_tokens=2048,
            stop=stop,
            n_ctx=3900,
            # callback_manager=callback_manager,
            verbose=False,
            # Verbose is required to pass to the callback manager
        )