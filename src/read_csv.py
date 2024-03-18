import os
from datasets import Dataset
import pandas as pd

from langchain.prompts import PromptTemplate
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from src.llama_writer import formatted_prompt_template

from trl import SFTTrainer

default_imdb_title_path = "./datas/title.basics.tsv"


def formatted_train_template():
    return PromptTemplate.from_template(
        """<|system|>
        {question}</s>
        <|assistant|>
        {response}</s>
        """
    )


class ImdbReader:
    def __init__(self, file_path=default_imdb_title_path):
        self.file_path = file_path

        if not os.path.exists(self.file_path):
            print("The file does not exist.")
            raise FileNotFoundError

        pd.options.mode.copy_on_write = True
        df = pd.read_csv(
            self.file_path,
            delimiter="\t",
            converters={"isAdult": lambda x: int(
                x) if isinstance(x, int) else 0},
        )
        # df["genre"] = df["genres"].str.split(",")
        # df = df.explode("genre")
        self.df = df


class TrainCSV:
    def __init__(self, model, df):
        self.model = model
        self.df = df.head().copy()

        # self.df["text"] = self.df.apply(self.format_row, axis=1)
        self.df.loc[:, "text"] = self.df.apply(self.format_row, axis=1)
        self.dataset = Dataset.from_pandas(self.df)

    def format_row(self, row):
        question = "Genre: {genre}, Movie: {title}, Year: {year}".format(
            genre=row["genres"],
            title=row["originalTitle"],
            year=row["startYear"],
        )

        # print("q ", question, row["originalTitle"])
        return formatted_train_template().format(
            question=question, response=row["originalTitle"]
        )

    def read(self):
        print(self.df["text"])
