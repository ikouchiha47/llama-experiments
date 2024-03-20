# csv.ai

It lets upload upto 1Gb CSV record, train and talk with it.

This works on CPU. And can be extended to GPU

## Requirements
- python 3.10+


## Tech used

- WordEmbedding model from hugging face for sentence transformer
- FAISS vector database
- Apple M1 16GB


## Running

```bash
make setup
make download.models # optional
make run
```

