from langchain_community.llms import LlamaCpp
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

from huggingface_hub import hf_hub_download
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
import torch

from transformers import AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig, CodeLlamaTokenizer, pipeline

from transformers import TapexTokenizer, BartForConditionalGeneration

from langchain_community.llms import HuggingFacePipeline


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


class CodeLlamaModel:
    model_name = "TheBloke/CodeLlama-7B-GGUF"
    t_model_name = "codellama/CodeLlama-7b-Instruct-hf"
    model_file = "codellama-7b.Q5_K_S.gguf"


class TapexModel:
    t_model_name = "microsoft/tapex-large-finetuned-wtq"
    model_name = "microsoft/tapex-large-finetuned-wtq"
    model_file = None


class LanguageModelLoader:
    # @suppress_error_output
    model = TapexModel()

    def __init__(self, model_name=None, model_file=None):
        self.model_name = self.model.model_name if model_name is None else model_name
        self.model_file = self.model.model_file if model_file is None else model_file
        self.verbose = False

        self.model_path = hf_hub_download(self.model_name, filename=self.model_file, cache_dir="./models/")
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        callback_manager = [FinalStreamingStdOutCallbackHandler()]

        self.model = LlamaCpp(
            model_path=self.model_path,
            tokenizer=self.model_path,
            temperature=0,
            callbacks=callback_manager,
            n_batch=100,
            n_gpu_layers=0,
            streaming=True,
            max_tokens=2048,
            n_ctx=4096,
            # context_window=4096,
            verbose=self.verbose,
        )
        # self.model = self.model.to_bettertransformer()


class LanguageModelLoaderGPU:
    # @suppress_error_output
    model_cfg = TapexModel()

    def __init__(self, model_name=None, model_file=None, df=None):
        self.model_name = self.model_cfg.model_name if model_name is None else model_name
        self.model_file = self.model_cfg.model_file if model_file is None else model_file
        self.verbose = False

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=100.0)

        _model = BartForConditionalGeneration.from_pretrained(
            self.model_cfg.t_model_name,
            torch_dtype=torch.float16,
            cache_dir="./models",
            # quantization_config=quantization_config,
        )
        _tokenizer = TapexTokenizer.from_pretrained(
            self.model_cfg.t_model_name,
            model_max_length=10007089,
            cache_dir="./models",
        )

        # print("paths ...... ", _model.name_or_path, _tokenizer.name_or_path)
        # self.model = LlamaCpp(
        #     model_path=_model.name_or_path,
        #     tokenizer=_tokenizer.name_or_path,
        #     temperature=0,
        #     # callbacks=callback_manager,
        #     n_batch=100,
        #     n_gpu_layers=0,
        #     streaming=True,
        #     max_tokens=2048,
        #     n_ctx=4096,
        # )
        #
        self.model = _model
        self.tokenizer = _tokenizer
