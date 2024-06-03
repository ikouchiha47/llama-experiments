# litkodeclassifier

A classifier for algorithmic questions.

## Requirements

- python3
- ~huggingface access (to download the GGUF)~
- ollama
- we will see

### Ollama

- Install from [https://ollama.com/download](https://ollama.com/download).
- Install llama3 using, `ollama run llama3`
- This should also start the inference server at `http://localhost:11434/`
- GreaseMonkey/TamperMonkey depending on firefox or chrome

You can check with `curl http://localhost:11434` or `sudo lsof -i :11434`


**Langchain Ollama Defaults**

- temperature: 0.8
- repeat_penalty: 1.1
- top_k: 40
- top_p: 0.9

## Data collection

There is a `fetcher.js` which has the code to get the data from the webpage using DOM Manipulation:

```js
function getTagName() {
    return window.location.pathname.
        replace("/tag/", "").
        replace("/", "").
        replace("-", "_")
  }

  function getProblems() {
    return Array.from($$(".title-cell__ZGos a")).map(el => el.innerText)
  }

  function getTitleSlug() {
    return $$(".title__PM_F")[0].innerText
  }

  function getPostData() {
    return {
      problems: getProblems(),
      tag_name: getTagName(),
      title: getTitleSlug()
    }
  }
```

### Running

Server setup:

```shell
# make sure ollama is running

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
./api.py
```

Frontend Setup

- Install GreaseMonkey/TamperMonkey
- Browser to any URL like `https://leetocde.com/tag/<whatever-valid-url>`
- Add a Grease/Tamper monket script.
- Copy Paste the code in `fetcher.js`.
- Save and Close, and reload the leetcode webpage, and run

```js
window.categorize(window.getPostData().title)
```
