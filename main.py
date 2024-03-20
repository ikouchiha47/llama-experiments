import sys
from src.llm import TinyLlm
from src.mammal import (
    TinyLamaUniverse,
    ConvesationBot,
)
from src.templates import ImdbChatPromptTemplate, IPLChatPromptTemplate

from src.dataloader import ImdbConfig, IPLConfig


Usage = """
python3 main.py [read, run]
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Invalid command")
        print(Usage)
        sys.exit(1)

    command = sys.argv[1]

    llm = TinyLlm()
    # cfg = ImdbConfig()
    cfg = IPLConfig()

    llama = TinyLamaUniverse(llm.model, cfg, cfg.vectordb_name)

    if command == "read":
        llama.read_tsv()
        llama.index_db(cfg.meta_keys)

    llama.load_vector_store_local()
    llama.build_qa(llama.vectorstore, IPLChatPromptTemplate())
    # llama.build_qa(llama.vectorstore, ImdbChatPromptTemplate())

    bot = ConvesationBot(llama.qa)

    while True:
        query = input("Input Prompt: ")
        if query == "exit":
            sys.exit(0)

        if query == "":
            continue

        bot.make_conversation(query, llama.retriever)
        print("")
        print("")
