const wrapper = document.querySelector(".wrapper"),
editableInput = wrapper.querySelector(".editable"),
readonlyInput = wrapper.querySelector(".readonly"),
placeholder = wrapper.querySelector(".placeholder"),
counter = wrapper.querySelector(".counter"),
button = wrapper.querySelector("button"),
classificationResult = wrapper.querySelector("#classification-result");

editableInput.onfocus = ()=>{
  placeholder.style.color = "#c5ccd3";
}
editableInput.onblur = ()=>{
  placeholder.style.color = "#98a5b1";
}

editableInput.onkeyup = (e)=>{
  let element = e.target;
  validated(element);
}
editableInput.onkeypress = (e)=>{
  let element = e.target;
  validated(element);
  placeholder.style.display = "none";
}

function validated(element){
  let text;
  let maxLength = 1000;
  let currentlength = element.innerText.length;

  if(currentlength <= 0){
    placeholder.style.display = "block";
    counter.style.display = "none";
    button.classList.remove("active");
  }else{
    placeholder.style.display = "none";
    counter.style.display = "block";
    button.classList.add("active");
  }

  counter.innerText = maxLength - currentlength;

  if(currentlength > maxLength){
    let overText = element.innerText.substr(maxLength); //extracting over texts
    overText = `<span class="highlight">${overText}</span>`; //creating new span and passing over texts
    text = element.innerText.substr(0, maxLength) + overText; //passing overText value in textTag variable
    readonlyInput.style.zIndex = "1";
    counter.style.color = "#e0245e";
    button.classList.remove("active");
  }else{
    readonlyInput.style.zIndex = "-1";
    counter.style.color = "#333";
  }
  readonlyInput.innerHTML = text; //replacing innerHTML of readonly div with textTag value
}

// add button onclick event listner, send the tweet to backend
button.onclick = (e)=>{
  classificationResult.innerText = ""
  fetch("http://localhost:8000/classify", {
    method: "POST",
    body: JSON.stringify({
      tweet: editableInput.innerText,
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8"
    }
  })
  .then((res) => {
    res.json()
    .then((json) => {
      console.log("Request complete! response:", json);
      classificationResult.innerText = json["type"]
    })
  })
};