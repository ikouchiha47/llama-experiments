from langchain_core.prompts import PromptTemplate
import sys
from src.llm import TinyLlm
from src.mammal import TinyLamaUniverse, ChatPromptTemplate, ConvesationBot

from typing import TypeVar

T = TypeVar("T", bound="ImdbConfig")


class ImdbConfig:
    def __init__(self):
        self.file_path = "../csv-reading-gpu/datas/title.basics.tsv"
        self.columns = ["originalTitle", "startYear", "genres"]
        self.sep = "\t"
        self.meta_keys = {
            "originalTitle": "str",
            "startYear": "str",
            "genres": "str",
            "text": "str",
        }

    def format_row(self, row):
        question = "genre: {genre}, movie: {title}, year: {year}".format(
            genre=row.genres,
            title=row.originalTitle,
            year=row.startYear,
        )

        return question


class IPLConfig:
    file_path = "./datasets/each_match_records.csv"
    columns = ["venue", "team1", "team2",
               "toss_won", "toss_decision", "winner"]
    sep = ","
    meta_keys = {
        "venue": "str",
        "team1": "str",
        "team2": "str",
        "toss_won": "str",
        "toss_decision": "str",
        "winner": "str",
        "text": "str",
    }

    template = (
        "Match was played between two teams {team1} and {team2} at {venue}."
        " {toss_won} won the toss and decided to {decision}."
        " {winner} won the match."
    )

    def format_row(self, row):
        prompt = PromptTemplate.from_template(self.template)

        data = prompt.format(
            team1=row.team1,
            team2=row.team2,
            venue=row.venue,
            toss_won=row.toss_won,
            decision=row.toss_decision,
            winner=row.winner,
        )
        return data


DataLoaderConfig = TypeVar("DataLoaderConfig", ImdbConfig, IPLConfig)

if __name__ == "__main__":
    llm = TinyLlm()
    # cfg = ImdbConfig()
    cfg = IPLConfig()

    llama = TinyLamaUniverse(llm, cfg, "ipl_db")

    llama.read_tsv()
    llama.index_db(cfg.meta_keys)

    # llama.load_vector_store_local()
    sys.exit(0)
    llama.build_qa(llama.vectorestore, ChatPromptTemplate())

    bot = ConvesationBot(llama.qa)

    while True:
        query = input("Input Prompt: ")
        if query == "exit":
            sys.exit(0)

        if query == "":
            continue

        print("Response ", bot.make_conversation(query)["answer"])
