document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("pre").forEach(function (pre) {
    const code = pre.querySelector("code");
    if (!code || pre.querySelector(".copy-button")) return;

    const button = document.createElement("button");
    button.className = "copy-button";
    button.type = "button";

    // Modern clipboard copy icon
    button.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"
           fill="currentColor" viewBox="0 0 24 24">
        <path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v16h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 18H8V7h11v16z"/>
      </svg>
    `;

    button.addEventListener("click", function () {
      navigator.clipboard.writeText(code.innerText).then(() => {
        button.innerHTML = "✅";
        setTimeout(() => (button.innerHTML = `
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
               fill="currentColor" viewBox="0 0 24 24">
            <path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v16h14a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 18H8V7h11v16z"/>
          </svg>
        `), 500);
      });
    });

    pre.appendChild(button);
  });
});
