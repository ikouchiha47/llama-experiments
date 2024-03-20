# from langchain_community.document_loaders import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings

import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import torch

import os
from pathlib import Path

from .templates import ChatTemplate

pd.set_option("display.max_colwidth", None)


class TinyLamaUniverse:
    def __init__(
        self,
        llm,
        cfg,
        vectorstore_name,
    ):
        self.llm = llm
        self.csv_file_path = cfg.file_path
        self.csv_sep = cfg.sep
        self.cfg = cfg
        self.vectorstore_path = f"./datastore/{vectorstore_name}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qa = None
        self.model = None
        self.merged_index = None
        self.models_path = Path("./models/sentence-transformers/all-MiniLM-L6-v2")

        self.__load_trasformer()

    def __load_trasformer(self):
        model_path = "sentence-transformers/all-MiniLM-L6-v2"  # remote
        if os.path.exists(str(self.models_path / "config.json")):
            model_path = str(self.models_path)

        SentenceTransformer(
            model_path,
            device=self.device,
        )

    def read_tsv(self):
        self.read_tsv_pd()

    def read_tsv_pd(self):
        df = dd.read_csv(
            self.csv_file_path,
            sep=self.csv_sep,
            # dtype=str,
            dtype=self.cfg.meta_keys,
            usecols=self.cfg.columns,
            blocksize=self.cfg.blocksize,
        )

        self.model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": False},
        )

        self.merged_index = FAISS.from_texts([""], self.model)

        # print(df.dtypes)
        # print(df.memory_usage(deep=True))

        self.df = df
        return self

    def __index_parition(self, partition):
        result = partition.apply(self.cfg.format_row, axis=1)
        index = FAISS.from_texts(result.tolist(), self.model)

        self.merged_index.merge_from(index)
        return result

    def index_db(self, meta_keys={}):
        if self.df is None:
            raise Exception("UninitializedDataframeException")

        if self.model is None:
            raise Exception("UninitializedModelExeception")

        if self.merged_index is None:
            raise Exception("UninitializedIndexException")

        with ProgressBar():
            partitions = self.df.map_partitions(
                self.__index_parition
                # , meta={"text": "str"}
            )

            partitions.compute()

        # partitions.visualize()
        self.merged_index.save_local(self.vectorstore_path)

    def load_vector_store_local(self):
        self.model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": False},
        )

        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.model,
            allow_dangerous_deserialization=True,
        )

    def build_qa(self, vectorstore, prompt: ChatTemplate):
        if vectorstore is None:
            raise Exception("UninitializedVectorStoreException")

        # memory = ConversationBufferMemory(
        #     memory_key="chat_history",
        #     return_messages=True,
        # )

        self.retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            return_source_documents=True,
            retriever=self.retriever,
            response_if_no_docs_found="I don't know",
            # condense_question_prompt=prompt.template,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt.template},
            # memory=memory,
        )


class ConvesationBot:
    def __init__(self, qa):
        self.qa = qa
        self.chat_history = []
        self.chat_history_file = "./datasets/chat_history.txt"

    def make_conversation(self, query, context):
        result = self.qa.invoke(
            {
                "question": query,
                "chat_history": self.chat_history,
            }
        )

        self.chat_history.extend([(query, result["answer"])])

        # self._write_chat_history_to_file()
        return result["answer"]

    def clear_chat(self):
        self.chat_history = []

    def _write_chat_history_to_file(self):
        with open(self.chat_history_file, "a") as f:
            for query, response in self.chat_history:
                f.write(f"User: {query}\n")
                f.write(f"Bot: {response}\n")
