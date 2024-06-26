import textwrap
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


class OneShotChat:
    def __init__(self):
        self.system_prompt = """\
        You are a helpful AI assistant, who can classify and categorize problem statements.
        You are given a list of {category} problems. Your task is to categorize by:
        - Identifying the similarities among problem statements.
        - Define clear and distinct classifications based on the identified similarities
        - Sub Group the problems into these defined classification.
        Answer the user query based on the provided problems statements."""

        self.user_prompt = """Question:
        Here are the list of {category} problem statements (separated by new lines):
        {problems}

        {question}."""

        prompt_template = textwrap.dedent(f"""
        {self.system_prompt}
        {self.user_prompt}""")

        self.prompt = PromptTemplate.from_template(prompt_template)
        self.parser = StrOutputParser()

    def partial(self, category, problems):
        self.prompt = self.prompt.partial(category=category, problems=problems)
        return self


from dataclasses import dataclass


@dataclass
class ExampleProblem:
    problem: str
    input: str
    output: str
    explain: str
    tags: str

    def to_prompt(self):
        prompt = [f"Problem Statement: {self.problem}"]
        if self.input:
            prompt.append(f"Example Input: {self.input}")
        if self.output:
            prompt.append(f"Expected Output: {self.output}")
        if self.explain:
            prompt.append(f"Explanation: {self.explain}")
        if self.tags:
            prompt.append(f"Tags: {self.tags}")

        return "\n".join(prompt)


class OneShotCode:
    def __init__(self, default_lang="Javascript"):
        self.system_prompt = textwrap.dedent("""
        You are an expert programmer that writes simple, concise, optimized code and concise explanations covering edge cases.
        Provide solutions for the Problem statement and efficient code in the programming language specified in the Question. In case nothing is specified use {default_lang}.
        - The code should have a valid syntax.
        - The code should be wrapped inside ``` and ```.
        - The code should have short 1 to 2 line comments where needed
        - Extract common functionality into separate functions.
        - Each function should do one thing and one thing well.
        - The answer should specify the time complexity.
        - Your explanation should summarize the solution and choice of data structure and algorithm.
        - The calculations using your approach must match the Expected Output for Example input if provided.
        - Do NOT explain step by step.
        - Do NOT makeup functions and libraries that are not present in the language.
        - Do NOT cite sources for solution.
        - Do NOT modify the Problem Statement.
        """).format(default_lang=default_lang)

        self.user_prompt = textwrap.dedent("""\
        {problem}
        {question}.""")

        prompt_template = f"""{self.system_prompt}\n{self.user_prompt}"""

        self.prompt = PromptTemplate.from_template(prompt_template)
        self.parser = StrOutputParser()

    def partial(
        self,
        exproblem: ExampleProblem,
    ):
        self.prompt = self.prompt.partial(problem=exproblem.to_prompt())
        return self
