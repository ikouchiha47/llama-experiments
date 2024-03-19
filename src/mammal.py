# from langchain_community.document_loaders import CSVLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import HuggingFaceEmbeddings

import faiss

import pandas as pd
import dask.dataframe as dd
import numpy as np
import torch

pd.set_option("display.max_colwidth", None)


class ChatPromptTemplate:
    def __init__(self):
        template = (
            "You are given the results of cricket matches played in the"
            "Indian Premier League(IPL) 2023.\n"
            "Combine the chat history and follow up question into "
            "a standalone question."
            "If you do not know the answer to a question,"
            "do not share false information."
            "Chat History: {chat_history}.\n"
            "Follow up question: {question}"
        )
        self.template = PromptTemplate.from_template(template)


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
        # self.model = SentenceTransformer(
        #     "sentence-transformers/all-MiniLM-L6-v2",
        #     device=self.device,
        # )

        self.model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": False},
        )
        print("herere 2, segfaults")

        self.merged_index = FAISS.from_texts([""], self.model)
        print("herere 3")

        # print(df.dtypes)
        # print(df.memory_usage(deep=True))

        self.df = df
        return self

    def __index_parition(self, partition):
        # db_ids = np.arange(len(partition)) + partition.index.start

        # result = partition.assign(text=self.cfg.format_row)
        # encoded_data = self.model.encode(
        #     result.tolist(),
        #     normalize_embeddings=False,
        #     show_progress_bar=True,
        # )
        #
        # index = faiss.IndexIDMap(
        #     faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        # )
        # #
        # faiss.normalize_L2(encoded_data)
        # index.add_with_ids(encoded_data, db_ids)

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

        partitions = self.df.map_partitions(
            self.__index_parition
            # , meta={"text": "str"}
        )

        # partitions.visualize()
        # print(partitions.head())
        partitions.compute()
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

        # self.vectorstore = FAISS(
        #     embeddings,
        #     faiss.read_index(self.vectorstore_path + "/index.faiss"),
        #     InMemoryDocstore({}),
        #     {},
        # )

    def build_qa(self, vectorstore, prompt: ChatPromptTemplate):
        if vectorstore is None:
            raise Exception("UninitializedVectorStoreException")

        # memory = ConversationBufferMemory(
        #     memory_key="chat_history",
        #     return_messages=True,
        # )
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            return_source_documents=True,
            retriever=vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 2}
            ),
            condense_question_prompt=prompt.template,
            # memory=memory,
        )


class ConvesationBot:
    def __init__(self, qa):
        self.qa = qa
        self.chat_history = []

    def make_conversation(self, query):
        result = self.qa.invoke(
            {"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        return result["answer"]

    def clear_chat(self):
        self.chat_history = []
