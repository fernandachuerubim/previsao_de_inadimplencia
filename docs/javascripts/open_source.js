window.addEventListener("load", function () {
  const summaries = document.querySelectorAll("summary");

  summaries.forEach((summary) => {
    if (summary.textContent.includes("Source code in")) {
      const details = summary.parentElement;
      if (details && details.tagName.toLowerCase() === "details") {
        details.open = true;
      }
    }
  });
});