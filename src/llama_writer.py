from langchain.prompts import PromptTemplate

# from llama_cpp import Llama
from src.llm_loader import TinyLlm


def formatted_prompt_template():
    return PromptTemplate.from_template(
        """<|system|>
        {system_message}</s>
        <|user|>
        {prompt}</s>
        <|assistant|>
        """
    )


class LlamaWriter:
    def __init__(self, model: TinyLlm):
        self.model = model

    def write(self, prompt):
        # prompt_template = f"""<|system|>
        # {self.model.system_message}</s>
        # <|user|>
        # {prompt}</s>
        # <|assistant|>
        # """
        prompt_template = formatted_prompt_template().format(
            system_message=self.model.system_message,
            prompt=prompt,
        )

        output = self.model.llm.invoke(
            prompt_template,
            max_tokens=256,
            stop=["</s>"],
            temperature=0.2,
            repeat_penalty=1.2,
            echo=False,
            # stream=True,
        )

        for text in output:
            if isinstance(text, str):
                yield text
                continue

            choices = text.get("choices")
            if choices:
                text = choices[0]["text"]
                yield text


# self.model = TinyLlm.new(model_name=model_name, model_file=model_file)
