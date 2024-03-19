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
import numpy as np
import torch


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
        csv_file_path,
        csv_delimiter=",",
    ):
        self.llm = llm
        self.csv_file_path = csv_file_path
        self.csv_sep = csv_delimiter
        self.vectorstore_path = "./datastore/titles_db"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def read_tsv(self):
        self.read_tsv_pd()

    def read_tsv_pd(self):
        pd.options.mode.copy_on_write = True

        df = pd.read_csv(
            self.csv_file_path,
            delimiter=self.csv_sep,
        )
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device=self.device,
        )
        df = df.copy()
        df.loc[:, "text"] = df.apply(self.format_row, axis=1)

        text_data = df.text.tolist()

        index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
        index = faiss.IndexIDMap(index)
        db_ids = df.index.values.astype(np.int64)

        # print(text_data, db_ids)

        encoded_data = model.encode(
            text_data, normalize_embeddings=False, show_progress_bar=True
        )
        # embeddings = []
        # batch_size = 1000
        # for i in range(0, len(text_data), batch_size):
        #     batch = text_data[i: i + batch_size]
        #     batch_embeddings = model.encode(
        #         batch, normalize_embeddings=False, show_progress_bar=True
        #     )
        #     embeddings.extend(batch_embeddings)
        #
        index.add(encoded_data, db_ids)
        # Save the FAISS index
        faiss.write_index(index, self.vectorstore_path)
        self.vectorestore = faiss

    def format_row(self, row):
        question = "genre: {genre}, movie: {title}, year: {year}".format(
            genre=row["genres"],
            title=row["originalTitle"],
            year=row["startYear"],
        )
        return question

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
