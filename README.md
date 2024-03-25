A different approach at csv processing.

## Tools:
- streamlit
- langchain
- openai, tinyllama or codellama (codellama might have some license issue)

Everything still runs on CPU.

## Approach

This time the approach is different. I have noticed, Pandas and SQL agents which
have a different way of handling prompts and querying the data.

For example the PandasQueryEngine keeps the dataframe in context. And produces a
prompt like so:

```
new_prompt = PromptTemplate(
"""\
You are working with a pandas dataframe in Python.
The name of the dataframe is `df`.
This is the result of `print(df.head())`:
{df_str}

Follow these instructions:
{instruction_str}
Query: {query_str}

Expression: """
```

The splitting and embedding manually is becoming a trouble. And it obviously
can't handle large files. Anyway it needs to become a separate job, handled by
a bunch of systems, like Spark jobs.

### File sizes less than 20MB
Use a `create_pandas_dataframe_agent` to pass the data.

### Larger file sizes
We read the columns, and create a table. Using dask we split the 
file and load the data into sqlite database. Use the `create_sql_agent`

[Reference for agent and agent executor](https://python.langchain.com/docs/modules/agents/concepts). In short the agent
uses the language model to choose a sequence of action to take. And then `agent executor` runs it.


## New requirements

Macos with metals gpu:

- conda

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
conda create -n llama python=3.10
conda activate llama
```

On silicon chips make sure metals can be run. Validate using:

```bash
xcrun metal
```

Incase it doesn't work, you need to install Xcode and then make `xcode-select
--print-path` point to `Xcode` 's path:

```
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

