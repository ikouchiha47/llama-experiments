from llm import ClosedAI
from prompt import OneShotChat
from trial import ProblemSet


prompter = OneShotChat()
gpt = ClosedAI()

ps = ProblemSet(
    "binary_search",
    "Binary Search",
    prompter,
    gpt,
    None,
)

ps.load_dataset("./problems/binary-search.txt")
