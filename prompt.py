from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


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
        self.prompt = prompt


class OneShot:
    def __init__(self):
        prompt_template = """
        Task: Group the following {category} problems by similarity and classify them into appropriate categories,
        ensuring each category has at most 10 problems. Follow the given instructions.

        Instructions:
        1. Carefully read the list of problems provided
        2. Identify the similarities among problems
        3. Define clear and distinct categories based on the identified similarities
        4. Group the problems into the defined categories, ensuring each category has at most 10 problems and at least 2.
        5. Review and refine the groups and categories for more accuracy.

        Problems:
        {problems}
        """

        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=["category", "problems"]
        )


class OneShotChat:
    def __init__(self):
        prompt_template = """
        <s>[INST] You are given a list of problems. You are to:
        1. Carefully read the list of problems provided
        2. Identify the similarities among problems
        3. Define clear and distinct classifications based on the identified similarities
        4. Group the problems into the defined classification, ensuring each category has at most 10 problems and at least 2.
        5. Review and refine the groups and categories for more accuracy.
        
        and answer the question based on the provided context

        Problems:
        {problems}

        Context:
        {context}

        Question: {question} [/INST]
        """

        self.prompt = ChatPromptTemplate.from_template(prompt_template)