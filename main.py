import sys
from src.llm import TinyLlm
from src.mammal import TinyLamaUniverse, ChatPromptTemplate, ConvesationBot

if __name__ == "__main__":
    llm = TinyLlm()
    llama = TinyLamaUniverse(
        llm,
        "../csv-reading-gpu/datas/title.basics.tsv",
        "\t",
    )
    llama.read_tsv()
    llama.index_db()
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
