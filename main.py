import sys

from src.llm import TinyLlm
from src.mammal import (
    CSVDocReader,
    ConversationBot,
)
from src.embedder import Embedder
from src.templates import ImdbChatPromptTemplate, IPLChatPromptTemplate
from src.view import ViewRunner

from src.dataloader import ImdbConfig, IPLConfig
import os

Usage = """
python3 main.py [read, run]
"""


def _validate_cli_args(args):
    if len(args) >= 2:
        return True

    print(Usage)
    sys.exit(1)


def start_embedding(_llm, _cfg):
    _llama = CSVDocReader(_llm.model, _cfg, _cfg.vectordb_name)

    file_size = os.path.getsize(_cfg.file_path)
    file_size_above_limit = True if file_size > 2e7 else False

    print("Using dask distributed client?", file_size_above_limit)

    _llama.read_tsv()
    embedder = Embedder(_llama.df, _cfg)
    embedder.index_db(use_dask_client=file_size_above_limit)


if __name__ == "__main__":
    _validate_cli_args(sys.argv)
    command = sys.argv[1]

    llm = TinyLlm()

    # cfg = ImdbConfig()
    cfg = IPLConfig()

    if command == "read":
        start_embedding(llm, cfg)
        print("reading and embedding complete")
        sys.exit(0)

    prompt_template = IPLChatPromptTemplate()
    # prompt_template = ImdbChatPromptTemplate()

    llama = CSVDocReader(llm.model, cfg, cfg.vectordb_name)
    bot = ConversationBot(llm.model, llama.vectorstore, prompt_template)

    if len(sys.argv) == 2:
        sys.argv[2] = "cli"

    if sys.argv[2] == "cli":
        ViewRunner.use_cli(bot)
    else:
        ViewRunner.use_st(bot)
