from typing import TypeVar

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType


class ImdbChatPromptTemplate:
    prompt_template = """
<s>[INST]<<SYS>>
You are given a list of movies, release date and genre.
When answering to user, if you do NOT know, just say that you do NOT know.
Do NOT make up answers from outside the context.

Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Context Information is below:
{context_str}\n\n

Given the new context and chat history, refine the original answer to better \
answer the next query.
Do NOT print the question.
<</SYS>>

<s>[INST]
User: {question}
[/INST]\n
Assistant:"""

    def __init__(self):
        self.template = PromptTemplate(
            self.prompt_template,
            prompt_type=PromptType.REFINE,
        )


class IPLChatPromptTemplate:
    prompt_template = """
<s>[INST]<<SYS>>
You are given the results of cricket matches played in the Indian \
Premier League(IPL) season 2023. The winner of IPL 2023 is the \
winner of the Final.

Your job as a data analyst is to understand user queries and \
respond with precise answers derived from the available information.

Use the provided context, chat history and instructions to guide \
your responses accurately.

When answering to user, if you do NOT know, just say that you do NOT know.
Do NOT make up answers from outside the context. Do not use external sources.

Avoid mentioning that you obtained the information from the context.
Answer according to the language of the user's question.

<</SYS>>

[INST] How many matches were played in IPL Season 2023? [/INST]
74

[INST] Who won the IPL 2023? [/INST]
Chennai Super Kings </s>

<s>[INST]
User: {question}
[/INST]\n
Assistant:"""

    def __init__(self):
        self.template = PromptTemplate(
            self.prompt_template,
            prompt_type=PromptType.REFINE,
        )


T = TypeVar("T", bound="ImdbChatPromptTemplate")

ChatTemplate = TypeVar(
    "ChatTemplate",
    ImdbChatPromptTemplate,
    IPLChatPromptTemplate,
)
