from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

from huggingface_hub import hf_hub_download
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


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


class TinyLlamaModel:
    model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    model_file = "tinyllama-1.1b-chat-v1.0.Q5_K_S.gguf"


class MistralModel:
    model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    model_file = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"


class LlamaChatModel:
    model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_file = "llama-2-7b-chat.Q5_K_S.gguf"


class CodeLlamaModel:
    model_name = "TheBloke/CodeLlama-7B-GGUF"
    model_file = "codellama-7b.Q5_K_S.gguf"


model = CodeLlamaModel()

tiny_model_name = model.model_name
tiny_model_file = model.model_file


class TinyLlm:
    # @suppress_error_output
    def __init__(self, model_name=tiny_model_name, model_file=tiny_model_file):
        self.model_name = model_name
        self.model_file = model_file
        self.verbose = False

        self.model_path = hf_hub_download(model_name, filename=model_file, cache_dir="./models/")
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        callback_manager = [FinalStreamingStdOutCallbackHandler()]

        self.model = LlamaCpp(
            model_path=self.model_path,
            temperature=0,
            callbacks=callback_manager,
            n_batch=100,
            n_gpu_layers=1,
            streaming=True,
            max_tokens=2048,
            n_ctx=2048,
            # context_window=4096,
            verbose=self.verbose,
        )

class OpenAILlm:
    def __init__(self):
        self.model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")