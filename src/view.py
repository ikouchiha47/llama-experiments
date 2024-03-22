import sys
import streamlit as st


class ViewRunner:
    @staticmethod
    def use_cli(bot):
        while True:
            query = input("Input Prompt: ")
            if query == "exit":
                sys.exit(0)

            if query == "":
                continue

            response = bot.make_conversation(query)
            bot.update_chat_history(["assistant", str(response)])
            print("Assistant: ", response)

    @staticmethod
    def use_st(bot):
        st.set_page_config(
            page_title="Build a RAGs bot, powered by LlamaIndex",
            page_icon="🦙",
            layout="centered",
            initial_sidebar_state="auto",
            menu_items=None,
        )
        st.title("Build a RAGs bot, powered by LlamaIndex 💬🦙")
        if (
            "messages" not in st.session_state.keys()
        ):  # Initialize the chat messages history
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "What do you want to know?",
                }
            ]

        def add_to_message_history(role: str, content: str) -> None:
            _message = {"role": role, "content": str(content)}
            st.session_state.messages.append(
                _message
            )  # Add response to message history

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # TODO: this is really hacky, only because st.rerun is jank
        if prompt := st.chat_input(
            "Your question",
        ):
            if "has_rerun" in st.session_state.keys() and st.session_state.has_rerun:
                st.session_state.has_rerun = False
            else:
                add_to_message_history("user", prompt)
                with st.chat_message("user"):
                    st.write(prompt)

                # If last message is not from assistant, generate a new response
                if st.session_state.messages[-1]["role"] != "assistant":
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            response = bot.make_conversation(prompt)
                            # response = current_state.builder_agent.chat(prompt)
                            st.write(str(response))
                            bot.update_chat_history(
                                ["assistant", str(response)])
                            add_to_message_history("assistant", str(response))
