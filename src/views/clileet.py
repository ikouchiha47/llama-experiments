from langchain.agents import AgentExecutor
from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationKGMemory,
    CombinedMemory, ChatMessageHistory,
    ConversationBufferMemory
)

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from prompt_toolkit import HTML, prompt, PromptSession
from prompt_toolkit.history import FileHistory
from langchain.agents.agent_types import AgentType
from langchain.input import get_colored_text
import pandas as pd

from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits.pandas.prompt import SUFFIX_WITH_DF

from ..llm import LanguageModelLoader


class PandasChatMemory:
    def __init__(self, llm):
        chat_history_buffer = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history_buffer",
            input_key="input"
        )

        history = ChatMessageHistory()
        chat_history_summary = ConversationSummaryMemory.from_messages(
            llm=llm,
            memory_key="chat_history_summary",
            input_key="input",
            chat_memory=history
        )

        chat_history_KG = ConversationKGMemory(
            llm=llm,
            memory_key="chat_history_KG",
            input_key="input",
        )

        self.memory = CombinedMemory(memories=[chat_history_buffer, chat_history_summary, chat_history_KG])


TEMPLATE = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.

Summary of the whole conversation:
{chat_history_summary}

Last few messages between you and user:
{chat_history_buffer}

Entities that the conversation is about:
{chat_history_KG}

Your answer should produce valid python code, without any syntax errors \
and should be executable by python_repl_ast.
You should use the tools below to answer the question posed of you:
"""


class CliViewer:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name).dropna()

        print("getting llm model")
        _llm = LanguageModelLoader()
        # _llm = OpenAILlm()
        # chat_memory = PandasChatMemory(_llm.model)
        #
        # print("initializing pandas agent")
        # self.agent = create_pandas_dataframe_agent(
        #     llm=_llm.model,
        #     df=self.df,
        #     # extra_tools=[PythonREPLTool()],
        #     # suffix=SUFFIX_WITH_DF,
        #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #     prefix=TEMPLATE,
        #     early_stopping_method='generate',
        #     max_iterations=5,
        #     include_df_in_prompt=True,
        #     agent_executor_kwargs={
        #         'handling_parsing_errors': "Check your output and make sure it conforms!",
        #         'memory': chat_memory.memory,
        #     },
        #     verbose=True,
        # )

        df = SmartDataframe(self.file_name, config={"llm": _llm.model})

        print("pandas agent loading complete")
        # session = PromptSession(history=FileHistory(".agent-history-file"))
        # while True:
        question = "How many matches did Chennai Super Kings win and lose?"

        print("invoking agent")
        # result = self.agent.invoke({"input": question}, callbacks=[StreamingStdOutCallbackHandler()])

        print("reesult, ", df.chat(question, "string"))
        # print(get_colored_text(result["output"], "green"))

        # question = session.prompt(
        #     HTML("<b>Type <u>Your question</u></b>  ('q' to exit): ")
        # )
        # if question.lower() == 'q':
        #     break
        # if len(question) == 0:
        #     continue
        # try:
        #     print(get_colored_text("Response: >>> ", "green"))
        #
        #
        # except Exception as e:
        #     print(get_colored_text(f"Failed to process {question}", "red"))
        #     print(get_colored_text(f"Error {e}", "red"))
