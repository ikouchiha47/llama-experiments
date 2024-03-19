from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer
import faiss

import pandas as pd
import dask.dataframe as dd
import numpy as np
import torch

pd.set_option("display.max_colwidth", None)


class ChatPromptTemplate:
    def __init__(self):
        template = (
            "Combine the chat history and follow up question into "
            "a standalone question."
            "If you do not know the answer to a question,"
            "do not share false information."
            "Chat History: {chat_history}"
            "Follow up question: {question}"
        )
        self.prompt = PromptTemplate.from_template(template)


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

    def read_tsv(self):
        self.read_tsv_pd()

    def read_tsv_pd(self):
        # pd.options.mode.copy_on_write = True

        df = dd.read_csv(
            self.csv_file_path,
            sep=self.csv_sep,
            dtype=str,
            usecols=self.cfg.columns,
            blocksize="128KB",
        )
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device,
        )

        self.merged_index = faiss.IndexIDMap(
            faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        )

        # print(df.dtypes)
        # print(df.memory_usage(deep=True))

        self.df = df
        return self

    def __index_parition(self, partition):
        db_ids = np.arange(len(partition))  # + partition.index.start

        result = partition.apply(self.cfg.format_row, axis=1)
        # result = partition.assign(text=self.cfg.format_row)
        encoded_data = self.model.encode(
            result.tolist(),
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        #
        index = faiss.IndexIDMap(
            faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        )
        #
        faiss.normalize_L2(encoded_data)
        index.add(encoded_data, db_ids)
        # FAISS.from_documents(result.tolist(), encoded_data)
        # self.merged_index.merge_from(index, partition.index.start)
        # result = partition
        # print(result.head())

        return result

    def index_db(self, meta_keys={}):
        if self.df is None:
            raise Exception("UninitializedDataframeException")

        if self.model is None:
            raise Exception("UninitializedModelExeception")

        if self.merged_index is None:
            raise Exception("UninitializedIndexException")

        # print(self.df.head(), meta_keys)
        partitions = self.df.map_partitions(
            self.__index_parition, meta={"text": "str"})

        # partitions.visualize()
        # print(partitions.head())
        partitions.compute()

    def read_tsv_slow(self):
        # need to handle windows-1252 encoding as well
        loader = CSVLoader(
            file_path=self.csv_file_path,
            encoding="UTF-8",
            csv_args={"delimiter": self.csv_sep},
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=10,
        )
        data = loader.load()
        text_chunks = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        vectorstore = FAISS.from_documents(text_chunks, embeddings)
        vectorstore.save_local(self.vectorstore_path)
        self.vectorestore = vectorstore

    def load_vector_store_local(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        FAISS.load_local(self.vectorstore_path, embeddings)

    def build_qa(self, vectorstore, prompt: ChatPromptTemplate):
        if self.qa is not None:
            return

        # question_generator_chain = LLMChain(llm=self.llm, prompt=prompt)

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 2}
            ),
            condense_question_prompt=prompt,
        )


class ConvesationBot:
    def __init__(self, qa):
        self.qa = qa
        self.chat_history = []

    def make_conversation(self, query):
        return self.qa({"question": query, "chat_hostory": self.chat_history})
