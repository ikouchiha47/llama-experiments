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
from pandasai import SmartDataframe

from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..llm import LanguageModelLoader, LanguageModelLoaderGPU


class PandasChatMemory:
    def __init__(self, llm):
        self.model = llm
        chat_history_buffer = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history_buffer",
            input_key="input"
        )
        self.memories = [chat_history_buffer]
        self.memory = chat_history_buffer

    def with_summary(self):
        history = ChatMessageHistory()
        chat_history_summary = ConversationSummaryMemory.from_messages(
            llm=self.model,
            memory_key="chat_history_summary",
            input_key="input",
            chat_memory=history
        )
        self.memories.append(chat_history_summary)
        return self

    def with_history(self):
        chat_history_KG = ConversationKGMemory(
            llm=self.model,
            memory_key="chat_history_KG",
            input_key="input",
        )
        self.memories.append(chat_history_KG)
        return self

    def get_memory(self):
        self.memory = CombinedMemory(memories=self.memories)
        return self.memory


from langchain import PromptTemplate

TEMPLATE = PromptTemplate.from_template("""
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.

Summary of the whole conversation:
chat_history_summary

Last few messages between you and user:
{chat_history_buffer}

Entities that the conversation is about:
chat_history_KG

Your answer should produce valid python code, without any syntax errors \
and should be executable by python_repl_ast.
You should use the tools below to answer the question posed of you:

python_repl_ast: A Python shell. Use this to execute python commands. \
Input should be a valid python command. When using this tool, \
sometimes output is abbreviated - make sure it does not look \
abbreviated before using it in your answer.

This is the result of `print(df.head())`:
{df_head}

Begin!
Question:
""")


class CliViewer:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name).astype(str)

        print("getting llm model", self.df.head())
        self.llm = LanguageModelLoaderGPU()

        question = "How many matches did Chennai Super Kings win and lose?"
        query = TEMPLATE.format(
            chat_history_buffer=[],
            df_head=self.df.head(),
            query=question,
        )
        encoding = self.llm.tokenizer(table=self.df, query=query, return_tensors="pt")
        outputs = self.llm.model.generate(**encoding)

        result = self.llm.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(result)

        # chat_memory = PandasChatMemory(_llm.model).with_history().with_summary()

    def __agen(self, _llm):
        chat_memory = PandasChatMemory(_llm.model).get_memory()
        self.agent = create_pandas_dataframe_agent(
            llm=_llm.model,
            df=self.df,
            # suffix=SUFFIX_WITH_DF,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=TEMPLATE,
            early_stopping_method='generate',
            max_iterations=5,
            include_df_in_prompt=True,
            agent_executor_kwargs={
                'handling_parsing_errors': "Check your output and make sure it conforms!",
                'memory': chat_memory,
            },
            verbose=True,
        )

        print("pandas agent loading complete", self.df.head())
        # session = PromptSession(history=FileHistory(".agent-history-file"))
        # while True:
        question = "How many matches did Chennai Super Kings win and lose?"

        print("invoking agent")
        result = self.agent.invoke(
            {"query": question, "table": self.df},
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        print("result", result)

        # print("reesult, ", df.chat(question, "string"))
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
