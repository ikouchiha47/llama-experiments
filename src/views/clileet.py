import pandas as pd
import json

from llama_index.core import Settings
from llama_index.core.query_engine import PandasQueryEngine

# from ..llm import TinyLlmGPU


class CliViewer:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name).astype(str)
        self.df["date"] = self.df["date"].apply(lambda x: "'" + x + "'")

        self.df = pd.DataFrame(
            {
                "city": ["Toronto", "Tokyo", "Berlin"],
                "population": [2930000, 13960000, 3645000],
            }
        )
        self.llm = Settings.llm
        self.query_engine = PandasQueryEngine(
            df=self.df, verbose=True, synthesize_response=True
        )

    def query(self, question=None):
        question = "How many matches did Chennai Super Kings win and lose?"
        question = "What is the city with the highest population?"
        response = self.query_engine.query(question)
        print(json.dumps(response))
