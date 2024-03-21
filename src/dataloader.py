from langchain_core.prompts import PromptTemplate
from llama_index.core.schema import Document

from typing import TypeVar

T = TypeVar("T", bound="ImdbConfig")


class ImdbConfig:
    file_path = "../csv-reading-gpu/datas/title.basics.tsv"
    blocksize = "12MB"
    vectordb_name = "imdb_db"
    columns = ["originalTitle", "startYear", "genres"]
    sep = "\t"
    meta_keys = {
        "originalTitle": "str",
        "startYear": "str",
        "genres": "str",
    }
    template = (
        "Movie {title}, was released in the year {year}, "
        "belonging to the {genre} genre"
    )
    template = """
    "Below is an instruction that describes a task, \
    paired with an input that provides further context.

    ### Instruction:
    Summarise the movie {title}.

    ### Input:
    {title} was released in the year {year}. It belongs to the {genre} genres.

    ### Output:
    {title} {genre} {year}
    """

    def format_row(self, row):
        prompt = PromptTemplate.from_template(self.template)
        data = prompt.format(
            genre=row.genres,
            title=row.originalTitle,
            year=row.startYear,
        )
        node = Document(
            text=data,
            metadata={
                "genre": row.genres,
                "title": row.originalTitle,
                "year": row.startYear,
            }
        )

        return node


class IPLConfig:
    file_path = "./datasets/each_match_records.csv"
    blocksize = "128KB"
    vectordb_name = "ipl_db"
    columns = [
        "season",
        "date",
        "match_number",
        "match_type",
        "venue",
        "team1",
        "team2",
        "toss_won",
        "toss_decision",
        "winner",
    ]

    sep = ","
    meta_keys = {
        "season": "int",
        "date": "str",
        "match_number": "int",
        "match_type": "str",
        "venue": "str",
        "team1": "str",
        "team2": "str",
        "toss_won": "str",
        "toss_decision": "str",
        "winner": "str",
    }

    template = (
        "Match {match_number} was played between two teams "
        "{team1} and {team2} at {venue}. It was a {match_type} match. "
        "{toss_won} won the toss and decided to {decision}. "
        "{winner} won the match and {loser} lost the match."
    )

    template = """
"Below is an instruction that describes a task, \
paired with an input that provides further context.

### Instruction:
Summarise the results of IPL season {season} match number {match_number}.
    
### Input:
Match was played between {team1} and {team2} at {venue} on {date} (dd-mm-yyyy).\
 It was a {match_type} match.
 {toss_won} won the toss and decided to {decision}.
    
### Output:
{winner} won the {match_type} match and {loser} lost the match.
"""

    #     template = """
    # Match {match_number} was played between {team1} and {team2} \
    # in IPL {season}.
    # The match was held at {venue} on {date} (dd-mm-yyyy).
    # {toss_won} won the toss and decided to {decision}. \
    # It was a {match_type} match. {winner} won the match {match_number}.</s>
    #     """

    #     template = """
    # <|prompt|>
    # Who played in match {match_number} in IPL {season}?</s>
    # <|assistant|>
    # Match number {match_number} was played between {team1} and {team2}.</s>
    #
    # <|prompt|>
    # Where was the match {match_number} played in IPL {season}?</s>
    # <|assistant|>
    # The match was held at {venue} on {date} (dd-mm-yyyy).</s>
    #
    # <|prompt|>
    # Which team won the match {match_number}?</s>
    # <|assistant|>
    # Match {match_number} was a {match_type} match. \
    # {toss_won} decided to {decision}, and {winner} won the match,\
    # {loser} lost the match</s>
    #     """
    def format_row(self, row):
        prompt = PromptTemplate.from_template(self.template)
        teams = {row.team1, row.team2}
        lost = teams - {row.winner}

        if not lost:
            return ""

        loser = lost.pop()
        data = prompt.format(
            season=row.season,
            date=row.date,
            match_number=row.match_number,
            match_type=row.match_type,
            team1=row.team1,
            team2=row.team2,
            venue=row.venue,
            toss_won=row.toss_won,
            decision=row.toss_decision,
            winner=row.winner,
            loser=loser,
        )
        # node = TextNode(
        #     text=data,
        #     metadata={
        #         "season": row.season,
        #         "date": row.date,
        #         "match_number": row.match_number,
        #         "match_type": row.match_type,
        #         "team1": row.team1,
        #         "team2": row.team2,
        #         "venue": row.venue,
        #         "toss_won": row.toss_won,
        #         "decision": row.toss_decision,
        #         "winner": row.winner,
        #         "loser": loser,
        #     }
        # )
        #
        node = Document(
            text=data,
            metadata={
                "season": row.season,
                "date": row.date,
                "match_number": row.match_number,
                "match_type": row.match_type,
                "team1": row.team1,
                "team2": row.team2,
                "venue": row.venue,
                "toss_won": row.toss_won,
                "decision": row.toss_decision,
                "winner": row.winner,
                "loser": loser,
            }
        )

        # print(data)
        return node


DataLoaderConfig = TypeVar("DataLoaderConfig", ImdbConfig, IPLConfig)
