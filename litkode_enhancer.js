// ==UserScript==
// @name     LitKodeClassifier
// @version  1
// @match    https://leetcode.com/tag/*
// @run-at      document-start
// ==/UserScript==

function $$(el) {
  return document.querySelectorAll(el)
}

function getTagName() {
  return window.location.pathname.replace("/tag/", "").replace("/", "").replace("-", "_")
}

function ProblemEnhancer() {
  
  function getProblemList() {
  	return Array.from($$(".reactable-data tr"))
	}

  var Problem = {idx: -1, value: undefined, marked: false, el: undefined }
  var newProblem = (args) => {
    return {...Problem, ...(args ||{})}
  }

  function indexProblemSet(problems) {
    return problems.map((problem, idx) => { 
      let el = problem.querySelector("td[label='#']");
      return newProblem({idx: idx, value: Number(el.innerText), el: el})
    })
  }

  function findProblem(savedData, args) {
     return savedData.find(data => data.idx == args.idx || data.value == args.value) 
  }


  function getSaveProblemSet(key, args) {
    let savedData = JSON.parse(localStorage.getItem(`leetcode_${key}`) || "[]")  
    if(!args) return savedData;

    return findProblem(savedData, args)
  }

  function saveProblemState(key, problem) {
    let savedData = getSaveProblemSet(key);

    // check if problem set is present, then update it
    // since, we are using a map, and not filtering it
    // the array index is the index itself
    let oldProblem = savedData[problem.idx];
    if(oldProblem) {
      savedData[problem.idx] = {...oldProblem, value: problem.value, marked: problem.marked}
    } else {
      savedData[problem.idx] = {idx: problem.idx, value: problem.value, marked: problem.marked}
    }

    localStorage.setItem(`leetcode_${key}`, JSON.stringify(savedData))
    return
  }


  // get the saved problem states
  // modify all the saved elements to have data-marked=true
  // false is either data-marked=false or data-marked not present
  // attach handler to update the data attribute
  // and add style accordingly
  
  function attachStylesAndHandlers(key, savedData, problemElms) {
    let indexedProblems = indexProblemSet(problemElms);
    let setStyle = (el) => {
      let style = `background: ${el.dataset.marked == "true" ? "#d1ffa3" : "white"}`
      el.style =style;  
    }

    problemElms.forEach((el, idx) => {
      console.log("idx", idx, el);
      
      el.addEventListener('click', () => {
          let marked = el.dataset.marked;
          marked = marked == "true" ? "false" : "true";
          el.dataset.marked = marked;

          let problem = indexedProblems[idx];
          problem.maked = Boolean(marked);

          saveProblemState(key, problem);

          setStyle(el);
        })
    })

    savedData.
      filter(data => indexedProblems[data.idx].value).
      map(data => {
        let problem = indexedProblems.find(p => p.value == data.value);
        let el = problem.el;

        console.log(el);

        el.dataset.marked = ''+(data.marked || false);
        return el
      })
  }


  function fireUp(key) {
    let problems = getSaveProblemSet(key);
    attachStylesAndHandlers(key, problems, getProblemList());
  }
  
  return {
    fireUp: fireUp,
    getProblemList: getProblemList,
  }
}

window.addEventListener("load",() => {
  let enhancer = new ProblemEnhancer();
  enhancer.fireUp(getTagName());
})
    
