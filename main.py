from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains import LLMChain


import streamlit as st

from streamlit_agent.llm import ClosedAI


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

        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return

        self.text += token
        self.container.markdown(self.text)


class Setup:
    model = None
    qa_chain = None
    retriever = None

    system_template = """\
You are a coding assistant, producing valid code in the language asked by Human.
Do not reframe the question and answer in the format of the question.
Wrap the code in three backticks.

----------------
"""

    def __init__(self, files):
        # Setup LLM and QA chain
        self.model = ClosedAI().llm

    def get_qa_chain(self, msgs):
        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=msgs, return_messages=True
        )

        messages = [
            SystemMessagePromptTemplate.from_template(self.system_template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        chat_prompt = ChatPromptTemplate.from_messages(messages)

        qa_chain = LLMChain(
            llm=self.model,
            memory=memory,
            verbose=False,
            prompt=chat_prompt,
        )

        self.qa_chain = RunnableWithMessageHistory(
            qa_chain,
            lambda session_id: msgs,
            input_messages_key="question",
            history_messages_key="history",
        )

        return self.qa_chain


@st.cache_resource(ttl="1h")
def setup():
    return Setup([])


st.set_page_config(page_title="KoderAss", page_icon="📖")
st.title("📖 KoderAssistant")

msgs = StreamlitChatMessageHistory()
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# view_messages = st.expander("View the message contents in session state")

# chain = q_prompt | ClosedAI().llm
# Render current messages from StreamlitChatMessageHistory

ctx = setup()

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    print(st.chat_message(avatars[msg.type]), msg)
    st.chat_message(avatars[msg.type]).write(msg.content)


# If user inputs a new prompt, generate and draw a new response
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    config = {"configurable": {"session_id": "any"}}

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        config["callbacks"] = [stream_handler]
        response = ctx.get_qa_chain(msgs).invoke(
            {"question": user_query},
            config,
        )
        print(response.content)
        # msgs.add_user_message(response)
