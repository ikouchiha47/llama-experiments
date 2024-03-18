import torch
from datasets import Dataset
import pandas as pd
import os

from langchain.prompts import PromptTemplate
from peft import LoraConfig

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from trl import SFTTrainer

default_imdb_title_path = "./sample_data/title.basics.tsv"


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
            converters={"isAdult": lambda x: int(x) if isinstance(x, int) else 0},
        )
        # df["genre"] = df["genres"].str.split(",")
        # df = df.explode("genre")
        self.df = df


class TrainCSV:
    def __init__(self, model_path):
        self.model_path = model_path

        self.output_model = "./datas/tinyllama-imdb-titles"
        self.output_model_checkpoint = f"{self.output_model}/checkpoint-250"
        self.peft_config = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        # self.build_model_and_tokenizer()

    def format_row(self, row):
        question = "genre: {genre}, movie: {title}, year: {year}".format(
            genre=row["genres"],
            title=row["originalTitle"],
            year=row["startYear"],
        )

        return formatted_train_template().format(
            question=question, response=row["originalTitle"]
        )

    def build_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            # bnb_8bit_use_double_quant=True,
            # bnb_8bit_quant_type="nf4",
            # bnb_8bit_compute_dtype=torch.bfloat16
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        self.model = model
        self.tokenizer = tokenizer

        return self

    def build_dataset(self, df):
        df = df.copy()
        df.loc[:, "text"] = df.apply(self.format_row, axis=1)
        pd.set_option("display.max_colwidth", None)
        print(df.head()["text"])

    def train(self, base_model, df):
        df = df.copy()
        df.loc[:, "text"] = df.apply(self.format_row, axis=1)
        self.df = df
        self.dataset = Dataset.from_pandas(df)

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
        trainer = SFTTrainer(
            model=base_model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=self.tokenizer,
            packing=False,
            max_seq_length=2048,
        )
        trainer.train()
        return self

    def build_trained_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(
            model,
            self.output_model_checkpoint,
            from_transformers=True,
            device_map="auto",
        )
        self.trained_model = peft_model.merge_and_unload()
        return self.trained_model

    def read(self):
        print(self.df["text"])
