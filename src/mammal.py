# from langchain_community.document_loaders import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import numpy as np
# from sqlalchemy import make_url

from sentence_transformers import SentenceTransformer
import pandas as pd
import dask.dataframe as dd

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.postgres import PGVectorStore

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding

from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.prompts.base import PromptTemplate
from typing import Any

import torch
import os
from pathlib import Path

from .templates import ChatTemplate

pd.set_option("display.max_colwidth", None)


class CSVDocReader:
    def __init__(
            self,
            llm,
            cfg,
            vectorstore_name,
    ):
        self.df = None
        # self.llm = llm
        self._csv_file_path = cfg.file_path
        self._csv_sep = cfg.sep
        self._models_path = Path("./models/models--sentence-transformers--all-MiniLM-L6-v2")

        self.cfg = cfg
        self.vectorstore_path = f"./datastore/{vectorstore_name}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vectorstore_name = vectorstore_name
        # Use a DataStore interface here

        model_path = "sentence-transformers/all-MiniLM-L6-v2"  # remote

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=model_path,
            cache_folder="./models/",
            device=self.device,
            normalize=False
        )
        # Settings.embed_model = FastEmbedEmbedding(
        #     "sentence-transformers/all-MiniLM-L6-v2",
        #     cache_dir="./models/",
        #     threads=10,
        # )
        Settings.llm = llm

        # if os.path.exists(str(self._models_path / "config.json")):
        #     model_path = str(self._models_path)

        transformer = SentenceTransformer(
            model_path,
            device=self.device,
        )

        store = PGVectorStore.from_params(
            database=os.environ.get("POSTGRES_DB", "vectordb"),
            host=os.environ.get("POSTGRES_DB_HOST", "localhost"),
            password=os.environ.get("POSTGRES_PASSWORD", "testpwd"),
            port="5433",
            user=os.environ.get("POSTGRES_USER", "testuser"),
            table_name=self.vectorstore_name,
            embed_dim=transformer.get_sentence_embedding_dimension(),
        )

        self.context = StorageContext.from_defaults(vector_store=store)
        self.vectorstore = VectorStoreIndex.from_vector_store(vector_store=store)

    def read_tsv(self):
        self.read_tsv_pd()

    def read_tsv_pd(self):
        df = dd.read_csv(
            self._csv_file_path,
            sep=self._csv_sep,
            dtype=self.cfg.meta_keys,
            usecols=self.cfg.columns,
            blocksize=self.cfg.blocksize,
        )

        self.df = df
        return self


class ConversationBot:
    chat_history = []

    def __init__(self, llm, vectorstore: VectorStoreIndex, prompt: ChatTemplate):
        self.retriever = vectorstore.as_retriever(
            vector_store_query_mode="default",
            similarity_top_k=2,
        )
        # self.retriever = retriever
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model="local",
            system_prompt="System prompt for RAG agent.",
        )
        response_synthesizer = CompactAndRefine(
            service_context=service_context,
            streaming=True
        )
        query_engine = RAGStringQueryEngine(
            retriever=self.retriever,
            response_synthesizer=response_synthesizer,
            llm=llm,
            qa_prompt=prompt.template,
            chat_history=self.chat_history,
        )
        self.qa = query_engine

    def make_conversation(self, query):
        return self.qa.query(query)

    def clear_chat(self):
        self.chat_history = []


class RAGStringQueryEngine(CustomQueryEngine):
    """RAG String Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer
    llm: Any
    qa_prompt: PromptTemplate
    chat_history = []
    streaming = True

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)

        context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            self.qa_prompt.format(context=context_str, question=query_str, chat_history=self.chat_history)
        )
        # print(response)

        return str(response)
