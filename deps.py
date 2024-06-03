from dataclasses import dataclass
import logging

# from langchain_community.embeddings import HuggingFaceEmbeddings
from llm import OllamaAI
from prompt import OneShotChat
from problemset import MockProblemSet, ProblemSet, SaveToFileIfHashChange


@dataclass
class ProblemData:
    tag_name: str
    title: str
    problems: list[str]


class ProblemSetNotInitialized(Exception):
    """ProblemSet not loaded"""


class Deps:
    def __init__(self) -> None:
        self.prompter = OneShotChat()
        self.gpt = OllamaAI()
        self.embeddings = self.gpt.embeddings
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="all-MiniLM-L6-v2",
        # )
        self.problems = {}

    def get_problem_set(self, tag: str) -> ProblemSet:
        ps = self.problems.get(tag, MockProblemSet())
        # print(
        #     "debug",
        #     ps,
        #     type(ps),
        #     ps.has_loaded(),
        #     ps.raw_dataset,
        # )
        if ps and not ps.has_loaded():
            raise ProblemSetNotInitialized

        return ps

    def add_problem_set(self, data: ProblemData):
        filename = f"./problems/{data.tag_name}.txt"
        try:
            ps = self.get_problem_set(data.tag_name)
        except ProblemSetNotInitialized:
            ps = ProblemSet(
                data.tag_name,
                data.title,
                self.prompter,
                self.gpt,
                self.embeddings,
            )
            SaveToFileIfHashChange(filename, data.problems)
        except Exception as e:
            # print(e)
            logging.exception("something else failed while finding existing ps")
            raise e

        try:
            logging.info("loading vectordb")
            # if not ps.load_vectordb():
            #     logging.info("setting vectordb")
            #     docs = ps.load_dataset(filename)
            #     ps.save_data(docs)
            # else:
            #     ps.set_dataset(filename)
            if not ps.is_dataset_loaded():
                ps.set_dataset(filename)

            ps.setup_inference_chain()
            self.problems[data.tag_name] = ps

            logging.info("Problem set added successfully")
            return True
        except Exception as e:
            # print(e)
            logging.exception("failed to add problemset")
            return False
