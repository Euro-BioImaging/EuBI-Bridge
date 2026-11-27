document.addEventListener("DOMContentLoaded", function () {
  const searchHeader = document.querySelector(".wy-side-nav-search > a");
  if (searchHeader && !searchHeader.querySelector(".site-title")) {
    const title = document.createElement("span");
    title.className = "site-title";
    title.innerText = "EuBI-Bridge";  // You can make this dynamic if needed
    searchHeader.appendChild(title);
  }
});
