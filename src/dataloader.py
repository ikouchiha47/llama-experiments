from langchain_core.prompts import PromptTemplate
from llama_index.core.schema import Document

from typing import TypeVar
import os
from .utils import get_hardware_device

T = TypeVar("T", bound="ImdbConfig")


class ImdbConfig:
    file_path = "../csv-reading-gpu/datas/title.basics.tsv"
    model_path = "sentence-transformers/all-MiniLM-L6-v2"  # remote

    blocksize = "25MB"

    torch_device = get_hardware_device()

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

    @staticmethod
    def db_config(vector_dimension=384):
        return {
            "database": os.environ.get("POSTGRES_DB", "vectordb"),
            "host": os.environ.get("POSTGRES_DB_HOST", "localhost"),
            "password": os.environ.get("POSTGRES_PASSWORD", "testpwd"),
            "port": "5433",
            "user": os.environ.get("POSTGRES_USER", "testuser"),
            "table_name": "imdb_db",
            "embed_dim": vector_dimension,
        }

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
            },
        )

        return node


class IPLConfig:
    file_path = "./datasets/each_match_records.csv"
    model_path = "thenlper/gte-small"  # remote
    # model_path = "sentence-transformers/sentence-t5-base"
    torch_device = get_hardware_device()

    blocksize = "1MB"

    vectordb_name = "ipl_db_2"
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

    template = """
"Below is an instruction that describes a task, \
paired with an input that provides further context.

### Instruction:
Summarise the results of IPL season {{season}} match number {{match_number}}.

### Input:
Match was played between {{team1}} and {{team2}} at {{venue}} on {{date}} (dd-mm-yyyy).\
 It was a {{match_type}} match.
{{toss_won}} won the toss and decided to {{decision}}.

### Output:
Match number {{match_number}}, {{match_type}} match, was won by {{winner}} and \
lost by {{loser}}.
{{winner}} won the {{match_type}} match and {{loser}} lost the \
{{match_type}} match.
{% if final_match == True %}
{{winner}} won the IPL season {{season}}.
{% endif %}
"""

    """
{
  "match_number": "{{match_number}}",
  "match_date": "{{match_date}}",
  "match_type": "{{match_type}}",
  "location": "{{venue}}",
  "toss_winner": "{{toss_won}}",
  "toss_decision": "{{decision}}",
  "match_winner": "{{winner}}",
  "loser": "{{loser}}",
}

### Output:
Match was played between {{winner}} and {{loser}} at {{venue}} \
on {{date}} (dd-mm-yyyy). It was a {{match_type}} match. \
{{toss_won}} won the toss and decided to {{decision}}.
{% if final_match == True %}
{{winner}} won the IPL season {{season}}
{% endif %}
    """

    template = """
Below is an instruction that describes a task, \
paired with an input that provides further context.

Date is of the format (dd-mm-yyyy).

### Instruction:
Results of IPL season {{season}} match number {{match_number}}.

### Input:
Match Number: {{match_number}}
Match Date: {{date}} (dd-mm-yyyy)
Match Type: {{match_type}}
Location: {{venue}}
Toss Winner: {{toss_won}}
Toss Winner Decision: {{decision}}
Match Winner: {{winner}}
Match Loser: {{loser}}
{% if final_match == True %}
Tournament Winner: {{ winner }}
{% endif %}
"""

    @staticmethod
    def db_config(vector_dimension=384):
        return {
            "database": os.environ.get("POSTGRES_DB", "vectordb"),
            "host": os.environ.get("POSTGRES_DB_HOST", "localhost"),
            "password": os.environ.get("POSTGRES_PASSWORD", "testpwd"),
            "port": "5433",
            "user": os.environ.get("POSTGRES_USER", "testuser"),
            "table_name": "ipl_db_2",
            "embed_dim": vector_dimension,
        }

    def format_row(self, row):
        prompt = PromptTemplate.from_template(self.template, template_format="jinja2")
        teams = {row.team1, row.team2}
        lost = teams - {row.winner}

        if not lost:
            return ""

        loser = lost.pop()
        final_match = row.match_type == "Final" or row.match_type == "final"

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
            final_match=final_match,
        )

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
            },
        )

        # print(data)
        return node


DataLoaderConfig = TypeVar("DataLoaderConfig", ImdbConfig, IPLConfig)
