from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, prompt


class Chatesque:
    def __init__(self):
        system_prompt_tmpl = """
        You are given a list of {category} problems. Your task is to classify the problems by similarity and classify them into appropriate categories, ensuring each category has at most 10 problems, using the following instructions.

        Instructions:
        1. Carefully read the list of problems provided
        2. Identify the similarities among problems
        3. Define clear and distinct categories based on the identified similarities
        4. Group the problems into the defined categories, ensuring each category has at most 10 problems and at least 2.
        5. Review and refine the groups and categories for more accuracy.
        """

        user_prompt_tmpl = """
        Problems:
        {problems}
        """

        input_variables = ["category", "problems"]
        messages = [("system", system_prompt_tmpl), ("user", user_prompt_tmpl)]
        prompt = ChatPromptTemplate.from_messages(
            messages,
        )
        prompt.input_variables = input_variables

        self.parser = StrOutputParser()
        self.prompt = prompt

    def partial(self):
        return self


class OneShot:
    def __init__(self):
        self.system_prompt = """
        Task: Group the following {category} problems by similarity and classify them into appropriate categories,
        ensuring each category has at most 10 problems. Follow the given instructions.

        Instructions:
        1. Carefully read the list of problems provided
        2. Identify the similarities among problems
        3. Define clear and distinct categories based on the identified similarities
        4. Group the problems into the defined categories, ensuring each category has at most 10 problems and at least 2.
        5. Review and refine the groups and categories for more accuracy."""

        self.user_prompt = """"Problems: {problems}

        Question: {question}
        """

        prompt_template = f"""{self.system_prompt}
        {self.user_prompt}
        """

        self.parser = StrOutputParser()
        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=["category", "problems"]
        )

    def partial(self):
        return self


class OneShotChat:
    def __init__(self):
        self.system_prompt = """
You are a helpful AI assistant, who can classify and categorize problem statements.
You are given a list of {category} problems. Your task is to categorize by:
- Identifying the similarities among problem statements.
- Define clear and distinct classifications based on the identified similarities
- Sub Group the problems into these defined classification.
Answer the user query based on the provided problems statements."""

        self.user_prompt = """Question:
Here are the list of problems statements (separated by new lines):
{problems}

{question}."""

        prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{self.system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{self.user_prompt}<|eot_id|>
"""

        prompt_template = f"""{self.system_prompt}
{self.user_prompt}
"""
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.parser = StrOutputParser()

    def partial(self, category, problems):
        self.prompt = self.prompt.partial(category=category, problems=problems)
        return self
