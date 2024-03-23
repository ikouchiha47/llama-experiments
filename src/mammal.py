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

from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer

from llama_index.core.indices.service_context import ServiceContext
from typing import List

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
        self._models_path = Path(
            "./models/models--sentence-transformers--all-MiniLM-L6-v2"
        )

        self.cfg = cfg
        self.vectorstore_path = f"./datastore/{vectorstore_name}"
        self.vectorstore_name = vectorstore_name

        # model_path = "sentence-transformers/all-MiniLM-L6-v2"  # remote

        print("device", self.cfg.torch_device)

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=cfg.model_path,
            cache_folder="./models/",
            device=self.cfg.torch_device,
            normalize=False,
        )
        # Settings.embed_model = FastEmbedEmbedding(
        #     "sentence-transformers/all-MiniLM-L6-v2",
        #     cache_dir="./models/",
        #     threads=10,
        # )
        Settings.llm = llm

        model_path = cfg.model_path
        if os.path.exists(str(self._models_path / "config.json")):
            model_path = str(self._models_path)

        transformer = SentenceTransformer(
            model_path,
            device=self.cfg.torch_device,
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
    chat_history: List = []

    def __init__(self, llm, vectorstore: VectorStoreIndex, prompt: ChatTemplate):
        self.retriever = vectorstore.as_retriever(
            vector_store_query_mode="default",
            similarity_top_k=0.2,
        )

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model="local",
            system_prompt="System prompt for RAG agent.",
        )
        # response_synthesizer = CompactAndRefine(
        #     service_context=service_context, streaming=True
        # )
        # query_engine = RAGStringQueryEngine(
        #     retriever=self.retriever,
        #     response_synthesizer=response_synthesizer,
        #     llm=llm,
        #     qa_prompt=prompt.template,
        # )

        memory = ChatMemoryBuffer.from_defaults(token_limit=2048)

        self.qa = CondenseQuestionChatEngine.from_defaults(
            query_engine=vectorstore.as_query_engine(),
            condense_question_prompt=prompt.template,
            chat_history=self.chat_history,
            service_context=service_context,
            verbose=False,
            memory=memory,
        )

    def update_chat_history(self, data):
        self.chat_history.extend([data])

    def get_chat_history(self):
        return self.chat_history

    def make_conversation(self, query):
        # response = self.qa.stream_chat(query)
        # for token in response.response_gen:
        #     yield token
        return self.qa.chat(query)

    def clear_chat(self):
        self.chat_history = []
