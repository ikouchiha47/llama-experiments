import sys
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from llm import CodellamaAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser

from prompt import ExampleProblem, OneShotCode
import re


def TextToCodeObj(text: str):
    res = re.split(r"-{4}(\w+)", text)
    if len(res) < 2:
        panic("invalid problem syntax")
    res = res[1:]
    d = {
        "input": "",
        "output": "",
        "explain": "",
        "tags": "",
    }
    for i in range(0, len(res) - 1, 2):
        d[res[i].lower()] = res[i + 1].strip().replace("\n", " ")

    ep = ExampleProblem(
        problem=d["problem"],
        input=d["input"],
        output=d["output"],
        explain=d["explain"],
        tags=d["tags"],
    )
    return ep


def panic(msg):
    import sys

    if msg:
        print(msg)
    sys.exit(1)


prompter = OneShotCode()

gpt = CodellamaAI()
filename = sys.argv[1]

with open(filename) as f:
    stmt = f.read().strip()


ep = TextToCodeObj(stmt)
prompter = prompter.partial(ep)
#
#
chain = (
    {
        "question": RunnablePassthrough(),
    }
    | prompter.prompt
    | gpt.model
    | prompter.parser
)

question = sys.argv[2]
# print(question)
#
print(prompter.prompt.format(question=question))
#
# print(chain.invoke(question))
for chunk in chain.stream(question):
    print(chunk, end="", flush=True)
