from typing import Any, List
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
import logging
import threading

from llm import ClosedAI, OllamaAI
from prompt import OneShotChat


class MockProblemSet:
    def __init__(self):
        self.raw_dataset = ""

    def has_loaded(self) -> bool:
        return False


class ProblemSet:
    db_name: str
    category_name: str
    prompter: OneShotChat
    gpt: OllamaAI
    embeddings: Embeddings
    raw_dataset: str

    _loading = True
    _dataset_loaded = False
    _vectordb = None
    _retriever = None
    _inferer: Any

    def __init__(self, db_name, category_name, prompter, gpt, embeddings):
        self.db_name = db_name
        self.category_name = category_name
        self.prompter = prompter
        self.gpt = gpt
        self.embeddings = embeddings
        self._lock = threading.Lock()
        self.raw_dataset = ""

    def has_loaded(self) -> bool:
        return not self._loading

    def is_dataset_loaded(self) -> bool:
        return self._dataset_loaded

    def set_dataset(self, raw_file):
        loader = TextLoader(raw_file)
        docs = loader.load()

        self.raw_dataset = docs[0].page_content
        self._dataset_loaded = True

    def load_dataset(self, raw_file):
        logging.debug("loading dataset")

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
        logging.debug("saving dataset")

        vectordb = FAISS.from_documents(
            [doc for doc in docs],
            self.embeddings,
        )
        vectordb.save_local("./vectorstore", self.db_name)
        self._vectordb = vectordb
        return self

    def load_vectordb(self):
        logging.debug("loading vector db")

        db = None
        try:
            db = FAISS.load_local(
                "./vectorstore",
                embeddings=self.embeddings,
                index_name=self.db_name,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            logging.exception("Error loading local db")
            db = None

        self._vectordb = db
        return db

    def build_retriever(self):
        logging.debug("building retriever")

        if not self._vectordb:
            raise Exception("DbNotFound")

        self._retriever = self._vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4}
        )
        return self._retriever

    def setup_inference_chain(self):
        logging.debug("setting up inference chain")

        # self.build_retriever()
        #
        # if not self._retriever:
        #     raise Exception("RetrieverNotConfigured")

        prompter = self.prompter.partial(self.category_name, self.raw_dataset)

        chain = (
            {
                "question": RunnablePassthrough(),
            }
            | prompter.prompt
            | self.gpt.model
            | prompter.parser
        )
        self._loading = False
        self._inferer = chain

    def infer(self, question):
        logging.debug(f"making inference for {question}")

        if not self._inferer and not self._loading:
            raise Exception("FuckFuckingzFuck")

        # replace with self._lock.acquire(timeout=5):
        return self._inferer.invoke(question)

    def stream(self, question):
        logging.debug(f"making inference for {question}")

        if not self._inferer and not self._loading:
            raise Exception("Inference chain not setup")

        with self._lock:
            for chunk in self._inferer.stream(question):
                yield chunk


def SaveToFileIfHashChange(filename: str, data: list[str]):
    fileData = "\n".join(data)
    with open(filename, "w") as f:
        f.write(fileData)


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
