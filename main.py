from langchain_core.prompts import (
    PromptTemplate,
)
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_core.output_parsers import StrOutputParser


# from langchain_core.runnables.history import RunnableWithMessageHistory

import streamlit as st

from streamlit_agent.llm import KoderModel, ClosedAI

st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

template = """
<s>[INST]
<<SYS>>
You are a coding assistant, producing valid python code.
Do not reframe the question and answer in the format of the question.
Wrap the code in three backticks.
Provide explanation for the answer.
<</SYS>>

[/INST]
User: {input}
Assistant:
"""

q_prompt = PromptTemplate.from_template(template=template)


chain = q_prompt | ClosedAI(KoderModel).llm | StrOutputParser()

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    config = {"configurable": {"session_id": "any"}}
    if prompt is not None:
        response = chain.invoke({"input": prompt})
        st.chat_message("ai").write(response)

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
