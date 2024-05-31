from typing import Any, List
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter

from llm import ClosedAI

from prompt import Chatesque, OneShot, OneShotChat


class ProblemSet:
    db_name: str
    category_name: str
    prompter: OneShotChat | Chatesque | OneShot
    gpt: ClosedAI
    embeddings: Embeddings
    raw_dataset: str

    __vectordb = None
    __retriever = None
    __inferer: Any

    def __init__(self, db_name, category_name, prompter, gpt, embeddings):
        self.db_name = db_name
        self.category_name = category_name
        self.prompter = prompter
        self.gpt = gpt
        self.embeddings = embeddings

    def load_vectordb(self):
        db = None
        try:
            db = FAISS.load_local(self.db_name, self.embeddings)
        except Exception as e:
            print(e)
            db = None

        self.__vectordb = db
        return db

    def load_dataset(self, raw_file):
        loader = TextLoader(raw_file)
        docs = loader.load()

        self.raw_dataset = docs[0].page_content

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=0,
        )
        return text_splitter.split_documents(docs)

    def save_data(self, docs: List[Document]):
        vectordb = FAISS.from_documents(
            [doc for doc in docs],
            self.embeddings,
        )
        self.__vectordb = vectordb
        self.build_retriever()

    def build_retriever(self):
        if not self.__vectordb:
            raise Exception("DbNotFound")

        self.__retriever = self.__vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
        )
        return self.__retriever

    def setup_inference_chain(self, raw_dataset: str):
        if not self.__retriever:
            raise Exception("RetrieverNotConfigured")

        chain = (
            {
                "context": self.__retriever,
                "question": RunnablePassthrough(),
                "problems": lambda _: raw_dataset,
                "category": lambda _: self.category_name,
            }
            | self.prompter.prompt
            | self.gpt.model
            | StrOutputParser()
        )
        self.__inferer = chain

    def infer(self, question):
        if not self.__inferer:
            raise Exception("FuckFuckingzFuck")

        return self.__inferer.invoke(question)


def SaveFileIfHashChange(filename: str):
    with open(filename) as f:
        problem_list = f.read().strip().split("\n")

    return problem_list


"""
/api/litkode/binary-search
{"db_name": "binary-search", "category": "Binary Search", data: ["str", "str"]}

# raw_dawg = SaveFileIfHashChange(f"./problems/raw/{db_name}")

ps = ProblemSet(
    db_name,
    category,
    prompter,
    gpt,
    haguembeddings,
)

if not ps.load_vectordb():
    ps.save_data(ps.load_dataset(filename))

ps.setup_inference_chain(raw_dawg)
"""
