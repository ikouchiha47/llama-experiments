import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain

from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from streamlit_agent.llm import ChatModel, ClosedAI


class StreamHandler(BaseCallbackHandler):
    def __init__(
            self,
            container: st.delta_generator.DeltaGenerator,
            initial_text: str = "",  # noqa: e501
    ):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        print(prompts[0], "retrival prompts", kwargs.get("run_id"))

        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


class Embedder:
    retriever = None

    def __init__(self, uploaded_files):
        print(self.retriever is None)

        if self.retriever is not None:
            return

        self.retriever = self.configure_retriever(uploaded_files)

    def configure_retriever(self, files):
        # Read documents
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            docs.extend(loader.load())

        # Split documents
        print("splitting")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        print("splits length ", len(splits))

        # Create embeddings and store in vectordb
        # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"
        )
        vectordb = FAISS.from_documents(splits, embeddings)
        print("embedding complete")

        # Define retriever
        _retriever = vectordb.as_retriever(
            search_type="mmr", search_kwargs={"k": 4, "fetch_k": 8}
        )

        return _retriever


class Setup:
    model = None
    qa_chain = None
    retriever = None

    system_template = """\
Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Your answer should have bullet points and should not exceed 500 words.

----------------
{context}"""

    def __init__(self, files):
        # Setup LLM and QA chain
        self.model = ClosedAI(model_conf=ChatModel).llm
        self.retriever = Embedder(files).retriever

    def get_qa_chain(self, msgs):
        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=msgs, return_messages=True
        )

        messages = [
            SystemMessagePromptTemplate.from_template(self.system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        chat_prompt = ChatPromptTemplate.from_messages(messages)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.model,
            chain_type="stuff",
            retriever=self.retriever,
            memory=memory,
            verbose=False,
            combine_docs_chain_kwargs={
                "prompt": chat_prompt
            })

        return self.qa_chain


@st.cache_resource(ttl="1h")
def setup(files):
    return Setup(files)


st.set_page_config(page_title="LangChain: Chat with Documents", page_icon="🦜")
st.title("🦜 LangChain: Chat with Documents")

print("reloading page")

uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()


# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
ctx = setup(uploaded_files)

if st.sidebar.button("Clear message history"):
    msgs.clear()
    st.cache_resource.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    print(st.chat_message(avatars[msg.type]), msg)
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = ctx.get_qa_chain(msgs).invoke(
            {"question": user_query}, {"callbacks": [retrieval_handler, stream_handler]}
        )
        print(response)
        msgs.add_user_message(response['answer'])
