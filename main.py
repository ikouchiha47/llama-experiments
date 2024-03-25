import sys

from src.views.streamleet import StreamleetView
from src.views.clileet import CliViewer
import argparse

from llama_index.core import Settings
from llama_index.llms.langchain import LangChainLLM
from src.llm import TinyLlmGPU


Settings.llm = LangChainLLM(llm=TinyLlmGPU().model)


def read_command(args):
    print(f"Reading from {args.file}")


def run_web_command(args):
    print("Running web application...")
    StreamleetView()


def run_cli_command(args):
    if args.file is None:
        print("file not provided")
        sys.exit(1)

    print("Running CLI application...")
    CliViewer(args.file).query()


def main():
    parser = argparse.ArgumentParser(description="CLI application")
    parser.add_argument("-f", "--file", help="File to read from")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    # 'read' command
    subparsers.add_parser("read", help="Read from a file")
    # 'run web' command
    run_parser = subparsers.add_parser("run", help="Interact with chat engine")
    run_parser.add_argument("-web", help="Run the web application", action="store_true")
    # 'run cli' command
    run_parser.add_argument("-cli", action="store_true", help="Run the CLI application")

    args = parser.parse_args()
    # print(args)

    if args.command == "read":
        read_command(args)
    elif args.command == "run" and args.web:
        run_web_command(args)
    elif args.command == "run" and args.cli:
        run_cli_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
