from typing import TypeVar

from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

class ImdbChatPromptTemplate:
    prompt_template = """
<<SYS>>
You are given a list of movies, release date and genre.
When answering to user, if you do NOT know, just say that you do NOT know.
Do NOT make up answers from outside the context.

Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Context Information is below:
{context}\n\n

Chat history is below:
{chat_history}\n\n

Given the new context and chat history, refine the original answer to better answer the query.
Do NOT print the question. 
<</SYS>>

[INST]
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
<<SYS>>
You are given the results of cricket matches played in the Indian \
Premier League(IPL) season 2023. The winner of the season is the \
winner of the final match.

When answering to user, if you do NOT know, just say that you do NOT know.
Do NOT make up answers from outside the context.

Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Context Information is below:
{context}\n\n

Chat history is below:
{chat_history}\n\n

Given the new context and chat history, refine the original answer to better answer the query.
Do NOT print the question. 
<</SYS>>

[INST]
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
