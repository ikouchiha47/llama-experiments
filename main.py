import sys
from src.llm import TinyLlm
from src.mammal import (
    TinyLamaUniverse,
    ConversationBot,
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
    cfg = ImdbConfig()
    # cfg = IPLConfig()

    llama = TinyLamaUniverse(llm.model, cfg, cfg.vectordb_name)

    if command == "read":
        llama.read_tsv()
        llama.index_db(cfg.meta_keys)
        sys.exit(0)

    # prompt_template = IPLChatPromptTemplate()
    prompt_template = ImdbChatPromptTemplate()

    # llama.load_vector_store_local()
    # llama.build_qa(llama.vectorstore, prompt_template)

    bot = ConversationBot(llama.llm, llama.vectorstore, prompt_template)

    while True:
        query = input("Input Prompt: ")
        if query == "exit":
            sys.exit(0)

        if query == "":
            continue

        bot.make_conversation(query)
