// ==UserScript==
// @name     LitKodeClassifier
// @version  1
// @match    https://leetcode.com/tag/*
// @run-at      document-start
// ==/UserScript==

function LitKodeClassifier() {
  function $$(el) {
  	return document.querySelectorAll(el)
  }
  
  function getTagName() {
    return window.location.pathname.replace("/tag/", "").replace("/", "").replace("-", "_")
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

  function makeRequest() {
    let data = getPostData()

    return fetch(`http://localhost:5000/api/litkode/ingest/${data.tag_name}`, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    }).then(d => d.text()).
      then(x => {
        console.log(`Response ${x}`)
        return x
      }).catch(e => console.error(e))
  }

  function getInference(query) {
    let tagName = getTagName()

    return fetch(
      `http://localhost:5000/api/litkode/infer/${tagName}?query=${query}`
      ).
      then(response => response.text()).
      then(x => ({data: x})).
      catch(e => ({error: e}))

  }
  
  let lastLogTime = 0;
  const logInterval = 100; // Time in ms to debounce

  const debouncedLog = (output) => {
    const now = Date.now();
    if (now - lastLogTime > logInterval) {
      console.clear();
      console.log(output);
      lastLogTime = now;
    }
  };

  async function getInferenceStream(query) {
    let tagName = getTagName()
	lastLogTime = 0;

    let response = await fetch(
      `http://localhost:5000/api/litkode/stream/${tagName}?query=${query}`
    )

    if (!response.ok) {
      console.error('Network response was not ok');
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let outputElement = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('Stream complete');
        break;
      }

      const chunk = decoder.decode(value, { stream: true });
      outputElement += chunk;
//       console.clear()
//       console.log(outputElement)
      	 debouncedLog(outputElement)
    }
    

  }

  async function categorize(tag) {
    let req = await makeRequest();

    query = `List all the categories and a couple of problem statements for the given {tag} problems`
    await getInferenceStream(query)
  }

  async function infer(query) {
    let req = await makeRequest();

    await getInferenceStream(query)
  }
  
  window.getPostData = getPostData
  window.categorize = categorize
  window.infer = infer
}


function addJS_Node(text, s_URL, funcToRun) {
    var D                                   = document;
    var scriptNode                          = D.createElement ('script');
    scriptNode.type                         = "text/javascript";
    scriptNode.id                           = "categorizer"
    if (text)       scriptNode.textContent  = text;
    if (s_URL)      scriptNode.src          = s_URL;
    if (funcToRun)  scriptNode.textContent  = '(' + funcToRun.toString() + ')()';

    var targ = D.getElementsByTagName ('head')[0] || D.body || D.documentElement;
    targ.appendChild (scriptNode);
}

addJS_Node(null, null, LitKodeClassifier)

