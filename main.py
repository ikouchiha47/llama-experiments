from src.llm_loader import TinyLlm
from src.llama_writer import LlamaWriter
from src.read_csv import ImdbReader, TrainCSV

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

if __name__ == "__main__":
    read = ImdbReader(file_path="./datas/title.basics.tsv")
    trainer = TrainCSV(model_path)
    trainer.build_dataset(read.df[:50])
    # trainer.train(trainer.model, read.df.head())
    # model = trainer.build_trained_model()

    # story_writer = LlamaWriter(trainer.model, trainer.tokenizer)
    # prompt = "List 10 Movies in action genre"

# print("eos_tokens", trainer.tokenizer.eos_token)
# for text in story_writer.write(prompt):
#   print(text, end="")
# for text in story_writer.write(prompt):
#     print(text, end="")
