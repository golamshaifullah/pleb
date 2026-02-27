document.addEventListener("DOMContentLoaded", () => {
  const wrapPleb = (el) => {
    if (!el || el.querySelector(".logo-pleb")) {
      return;
    }
    const text = el.textContent || "";
    const match = text.match(/^(pleb)(\b.*)$/i);
    if (!match) {
      return;
    }
    const rest = match[2] || "";
    el.innerHTML = `<span class="logo-pleb">${match[1]}</span>${rest}`;
  };

  const logoLinks = document.querySelectorAll(".logo a");
  for (const link of logoLinks) {
    wrapPleb(link);
  }

  const contentHeadings = document.querySelectorAll(
    "div.document h1, div.body h1, main h1"
  );
  for (const heading of contentHeadings) {
    wrapPleb(heading);
  }
});
