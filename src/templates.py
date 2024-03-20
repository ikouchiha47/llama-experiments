from typing import TypeVar
from langchain_core.prompts import (
    # ChatPromptTemplate,
    PromptTemplate,
    # MessagesPlaceholder,
)


class ImdbChatPromptTemplate:
    prompt_template = """
You are given a list of movies, release date and genre.
Given the chat history delimited by(<hs></hs>) and \
question which might reference context (delimited by <ctx></ctx>) \
in chat history (delimited by <hs></hs>), formulate an answer.\
Do NOT print the question.

When answering to user, if you do NOT know, just say that you do NOT know.
Do NOT make up answers from outside the context.

Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

<ctx>
{context}
</ctx>

<hs>
{chat_history}
</hs>

<|user|>
{question}</s>
<|assistant|>
    """

    def __init__(self):
        self.template = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question", "chat_history"],
        )


class IPLChatPromptTemplate:
    prompt_template = """
<|system|>
You are given the results of cricket matches played in the Indian
Premier League(IPL) in the year 2023. Given the chat history \
delimited by(<hs></hs>) and question which might \
reference context (delimited by <ctx></ctx>) \
in chat history (delimited by <hs></hs>), formulate an answer.\
Do NOT print the question.

When answering to user, if you do NOT know, just say that you do NOT know.
Do NOT make up answers from outside the context.

Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

<ctx>
{context}
</ctx>

<hs>
{chat_history}
</hs>

<|user|>
{question}</s>
<|assistant|>
        """

    def __init__(self):
        self.template = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question", "chat_history"],
        )


T = TypeVar("T", bound="ImdbChatPromptTemplate")

ChatTemplate = TypeVar(
    "ChatTemplate",
    ImdbChatPromptTemplate,
    IPLChatPromptTemplate,
)
