from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from llm import OllamaAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser

from prompt import OneShotChat


def panic():
    import sys

    sys.exit(1)


prompter = OneShotChat()

# gpt = ClosedAI()
# gpt = OllamaAI(prompter)
gpt = OllamaAI()
embeddings = gpt.embeddings


def configure_retriever():
    # print("embedding start")
    loader = TextLoader("./problems/binary-search.txt")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
    )
    splits = text_splitter.split_documents(docs)

    vectordb = FAISS.from_documents(
        [split for split in splits],
        embeddings,
    )
    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
    )

    # print("embedding end")
    return retriever


filename = "./problems/binary-search.txt"
title = "Binary Search"

with open(filename) as f:
    problem_list = f.read().strip().split(",")


problem_list = "\n".join(map(lambda x: x.strip('"'), problem_list))
prompter = prompter.partial(title, problem_list)


chain = (
    {
        "question": RunnablePassthrough(),
    }
    | prompter.prompt
    | gpt.model
    | prompter.parser
)


# retriever = configure_retriever()
# chain = (
#     {
#         "context": retriever,
#         "question": RunnablePassthrough(),
#         "problems": lambda _: problem_list,
#         "category": lambda _: "Binary Search",
#     }
#     | prompter.prompt
#     | gpt.model
#     | prompter.parser
# )

question = "List all the categories and a couple of problem statements for the given Binary Search problems"
print(question)

print(prompter.prompt.format(question=question))

print(chain.invoke(question))
# , stop=["<|eot_id|>"]))
