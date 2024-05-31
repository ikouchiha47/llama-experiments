from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from llm import ClosedAI
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from prompt import OneShotChat


def panic():
    import sys

    sys.exit(1)


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
)


def configure_retriever():
    print("embedding start")
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

    print("embedding end")
    return retriever


with open("./problems/binary-search.txt") as f:
    problem_list = f.read().strip().split(",")


problem_list = "\n".join(map(lambda x: x.strip('"'), problem_list))
prompter = OneShotChat()


gpt = ClosedAI()

# ## try chain
# chain = prompter.prompt | gpt.model

retriever = configure_retriever()


chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "problems": lambda _: problem_list,
        "category": lambda _: "Binary Search",
    }
    | prompter.prompt
    | gpt.model
    | StrOutputParser()
)


chain.invoke("List all the problems and sub categories of Binary Search problems")
