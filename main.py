from src.llm_loader import TinyLlm
from src.llama_writer import LlamaWriter
from src.read_csv import ImdbReader, TrainCSV


if __name__ == "__main__":
    read = ImdbReader()
    model = TinyLlm()
    trained = TrainCSV(model, read.df)
    trained.read()

    story_writer = LlamaWriter(model)
    prompt = "Write a story about llamas"

    # for text in story_writer.write(prompt):
    #     print(text, end="")
