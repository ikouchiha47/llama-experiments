from langchain.prompts import PromptTemplate

from transformers import GenerationConfig


story_writer_system_message = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 
If you don't know the answer to a question, please don't share false information.

"""


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
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def write(self, prompt):
        prompt_template = formatted_prompt_template().format(
            system_message=story_writer_system_message,
            prompt=prompt,
        )

        generation_config = GenerationConfig(
            penalty_alpha=0.6,
            do_sample=True,
            top_k=40,
            temperature=0.5,
            # repetition_penalty=1.2,
            max_new_tokens=100,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        inputs = self.tokenizer([prompt_template], return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, generation_config=generation_config)

        yield self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def write_l(self, prompt):
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
