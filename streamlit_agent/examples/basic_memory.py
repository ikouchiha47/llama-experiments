from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import streamlit as st

from streamlit_agent.llm import ClosedAI

st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a coder who produces very efficient code."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


chain = prompt | ClosedAI().llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
