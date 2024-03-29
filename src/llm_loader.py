# from langchain.prompts import PromptTemplate
# from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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

story_writer_system_message = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information.

"""

CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])


class TinyLlm:
    def __init__(self, model_name=tiny_model_name, model_file=tiny_model_file):
        self.model_name = model_name
        self.model_file = model_file
        self.model_path = hf_hub_download(model_name, filename=model_file)
        self.verbose = False
        self.system_message = story_writer_system_message

        with suppress_stdout_stderr():
            self.llm = LlamaCpp(
                model_path=self.model_path,
                n_ctx=512,
                n_batch=10,
                n_threads=8,
                n_gpu_layers=0,
                callback_manager=CallbackManager,
                temperature=0.7,
                verbose=self.verbose,
            )
