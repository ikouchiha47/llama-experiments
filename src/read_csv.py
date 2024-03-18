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
        self.output_model = "./datas/tinyllama-imdb"
        self.df = df.copy()

        # self.df["text"] = self.df.apply(self.format_row, axis=1)
        self.df.loc[:, "text"] = self.df.apply(self.format_row, axis=1)
        self.dataset = Dataset.from_pandas(self.df)
        self.peft_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )

    def format_row(self, row):
        question = "###Genre### {genre} ###Movie### {title} ###Year#### {year}".format(
            genre=row["genres"],
            title=row["originalTitle"],
            year=row["startYear"],
        )

        # print("q ", question, row["originalTitle"])
        return formatted_train_template().format(
            question=question, response=row["originalTitle"]
        )

    # WIP
    def train(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto"
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        training_arguments = TrainingArguments(
            output_dir=self.output_model,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
            num_train_epochs=3,
            max_steps=250,
            fp16=True,
        )
        self.trainer = SFTTrainer(
            model=model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=1024,
        )
        self.trainer.train()
        return self

    def

    def read(self):
        print(self.df["text"])
