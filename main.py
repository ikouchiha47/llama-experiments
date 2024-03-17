from llama_cpp import Llama
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

story_writer_system_message = "You are a story writing assistant"


class LlamaWriter:
    def __init__(self, model_name=tiny_model_name, model_file=tiny_model_file):
        self.model_name = model_name
        self.model_file = model_file
        self.model_path = hf_hub_download(model_name, filename=model_file)
        self.verbose = False

        with suppress_stdout_stderr():
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=512,
                n_threads=8,
                n_gpu_layers=0,
                verbose=self.verbose,
            )

    def write(self, prompt, system_message=story_writer_system_message):
        prompt_template = f"""<|system|>
        {system_message}</s>
        <|user|>
        {prompt}</s>
        <|assistant|>
        """

        output = self.llm(
            prompt_template,
            max_tokens=256,
            stop=["</s>"],
            temperature=0.2,
            repeat_penalty=1.2,
            echo=False,
            stream=True,
        )

        for text in output:
            if isinstance(text, str):
                # print(text, end="")
                yield text
                continue

            choices = text.get("choices")
            if choices:
                text = choices[0]["text"]
                yield text
                # print(text, end="")


if __name__ == "__main__":
    story_writer = LlamaWriter()
    prompt = "Write a story about llamas"

    for text in story_writer.write(prompt):
        print(text, end="")
