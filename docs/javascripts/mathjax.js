window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: "[^\\s]+",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  if (typeof window.MathJax?.typesetPromise === "function") {
    window.MathJax.typesetPromise();
  }
});
