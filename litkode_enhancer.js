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

let MAX_LOOP = 20

function waitTill(fn, comp, cb) {
	let loopCount = 0;
  let interval = setInterval(() => {
    let [values, ok] = comp(fn())
    // console.log(values, ok, "running")
  
  	if(ok) {
      // console.log("result received")
      clearInterval(interval)
      cb(null, values);
      return
    }
    
    if(loopCount == MAX_LOOP) {
      // console.log("exit")
    	clearInterval(interval);
      cb("max loop", values);
      return
    }
    
    loopCount += 1;
    
  }, 1000)
}

const LeetcodeData = {
  problems: [],
  other: {}
}

function ProblemEnhancer() {
  let store = {...LeetcodeData}
  
  function getProblemList() {
  	return Array.from($$(".reactable-data tr"))
	}

  var Problem = {idx: -1, value: undefined, marked: false, el: undefined };
  
  var newProblem = (args) => {
    return {...Problem, ...(args ||{})}
  }

  function indexProblemSet(problems) {
    return problems.map((problem, idx) => { 
      let el = problem.querySelector("td[label='#']");
      return newProblem({idx: idx, value: Number(el.innerText), el: el});
    })
  }

  function findProblem(savedData, args) {
     return savedData.find(data => data.idx == args.idx || data.value == args.value) 
  }


  function getSaveProblemSet(key, args) {
    if(store.problems.length > 0) return store;
    
    let savedData = JSON.parse(localStorage.getItem(`leetcode_${key}`) || `{"problems": "[]", "other": "{}"}`)
    if(!args) return {...LeetcodeData, ...savedData};
    
    store = savedData;

    return findProblem(savedData.problems, args)
  }

  function saveProblemState(key, problem) {
    let savedData = getSaveProblemSet(key);

    // check if problem set is present, then update it
    // since, we are using a map, and not filtering it
    // the array index is the index itself
    let oldProblem = savedData.problems[problem.idx];
    if(oldProblem) {
      savedData.problems[problem.idx] = {...oldProblem, value: problem.value, marked: problem.marked}
    } else {
      savedData.problems[problem.idx] = {idx: problem.idx, value: problem.value, marked: problem.marked}
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
      // console.log("idx", idx, el);
      
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
      filter(data => data && indexedProblems[data.idx] && indexedProblems[data.idx].value).
      map(data => {
        let problem = indexedProblems.find(p => p.value == data.value);
        let el = problem.el;

        //console.log(el);

        el.dataset.marked = ''+(data.marked || false);
        return el
      })
  }


  function fireUp(key) {
    let data = getSaveProblemSet(key);
    let problems = data.problems;
    
    waitTill(getProblemList, (values) => [values, values && values.length > 0], (err, values) => {
      if(err == null)
      	attachStylesAndHandlers(key, problems, values);
      else
        console.log("error", err)
    });
    
  }
  
  function getDifficulty(key) {
    let data = getSaveProblemSet(key);
    let prevDifficulty = (data.other && data.other.difficulty) || "none";
    
    let diffcultClasses = [...$("th.reactable-th-difficulty").classList];
    let difficulty = diffcultClasses.find(el => el.match("desc")) || diffcultClasses.find(el => el.match("asc")) || "none";
    
    return [prevDifficulty, difficulty];
  }
  
  
  function storeDifficulty(key, difficulty) {
    let data = getSaveProblemSet(key);
    data.other.difficulty = difficulty;
    
    localStorage.setItem(`leetcode_${key}`, JSON.stringify(data))    
  }
  
  
  function handleDifficulty(e, key) {
  	let target = e.target
    
    if(!target) return;
    
    let [prev, present] = getDifficulty(key);
    if(present == "none") present="desc"
    
    if(prev != present) storeDifficulty(key, present);
  }
  
  function setPrevDifficulty(el, key) {
  	let [prev, present] = getDifficulty(key);
    if(present != "none" || prev == "none" || prev == present) {
    	return
    }
    
    // toggle asc desc
    if(prev == "desc") {
    	el.click()
    } else if(prev == "asc") {
    	el.click();
      setTimeout(() =>  el.click(), 2000);
    }
 	}
  
  function trackDifficulty(key) {
  	waitTill(
      () => {
      	return $("th.reactable-th-difficulty")
      },
      (el) => ([el, !!el]),
      (err, el) => {
      	if(err == null) {
          $("th.reactable-th-difficulty").addEventListener('click', (e) => handleDifficulty(e, key));
          setPrevDifficulty($("th.reactable-th-difficulty"), key)
        }
        else
          console.log("error ", err)
      },
    )
  }
  
  return {
    fireUp: fireUp,
    getProblemList: getProblemList,
    trackDifficulty:trackDifficulty,
  }
}

window.addEventListener("load",() => {
  let enhancer = new ProblemEnhancer();
  enhancer.fireUp(getTagName());
  enhancer.trackDifficulty(getTagName());
})
    
