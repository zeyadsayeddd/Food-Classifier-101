const $ = id => document.getElementById(id);
const analyze = $("analyze");
const imageName = $("image-name");
let selected = { src: $("preview").src };
let worker, objectUrl, revision = 0, busy = false;

function ready() {
  busy = false;
  analyze.disabled = false;
  analyze.querySelector("span").textContent = "Identify dish";
}
function select(photo) {
  revision++;
  $("announcement").textContent = "";
  if (busy) {
    worker?.terminate();
    worker = null;
  }
  ready();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = photo.file ? URL.createObjectURL(photo.file) : null;
  selected = photo;
  $("preview").src = objectUrl || photo.src;
  $("preview").alt = photo.name;
  imageName.textContent = photo.name;
  $("source").textContent = photo.file ? "Your photo" : "Sample photo";
  for (const button of $("samples").children) button.setAttribute("aria-pressed", String(button === photo.button));
  $("results").hidden = true;
  $("setup").hidden = false;
  $("change").hidden = true;
  analyze.hidden = false;
  $("notice").textContent = "";
}
function accept(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    $("notice").textContent = "Choose a JPG, PNG, or WebP image.";
    return;
  }
  select({ file, name: file.name });
}
for (const button of $("samples").children) button.onclick = () => select({
  button, name: button.dataset.name, src: button.querySelector("img").src,
});
$("choose").onclick = $("change").onclick = () => $("file").click();
$("file").onchange = e => { accept(e.target.files[0]); e.target.value = ""; };
for (const el of [$("frame"), document.querySelector(".controls")]) {
  el.ondragover = e => { e.preventDefault(); $("choose").classList.add("drag"); };
  el.ondragleave = () => $("choose").classList.remove("drag");
  el.ondrop = e => {
    e.preventDefault();
    $("choose").classList.remove("drag");
    accept(e.dataTransfer.files[0]);
  };
}
function showMatches({ matches, seconds }) {
  const rows = matches.map(({ label, probability }) => {
    const row = document.createElement("li");
    row.className = "match";
    const title = document.createElement("strong");
    title.textContent = (label[0].toUpperCase() + label.slice(1)).replaceAll("_", " ");
    const score = document.createElement("span");
    score.textContent = (probability * 100).toFixed(1) + "%";
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("i");
    fill.style.width = probability * 100 + "%";
    bar.append(fill);
    row.append(title, score, bar);
    return row;
  });
  $("ranking").replaceChildren(...rows);
  $("elapsed").textContent = seconds.toFixed(2) + " s";
  $("results").hidden = false;
  $("setup").hidden = true;
  analyze.hidden = true;
  $("change").hidden = false;
  $("announcement").textContent = "Best match: " + rows[0].querySelector("strong").textContent;
}
function reportError(message) {
  ready();
  $("notice").textContent = message;
}
function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.status) analyze.querySelector("span").textContent = data.status;
    else if (data.error) reportError(data.error);
    else { ready(); showMatches(data); }
  };
  worker.onerror = () => {
    worker?.terminate();
    worker = null;
    reportError("Couldn’t start the classifier. Check your connection and retry in a current browser.");
  };
  return worker;
}
analyze.onclick = async () => {
  if (busy) return;
  const current = revision;
  busy = true;
  analyze.disabled = true;
  analyze.querySelector("span").textContent = "Preparing photo…";
  $("notice").textContent = "";
  try {
    let file = selected.file;
    if (!file) {
      const response = await fetch(selected.src);
      if (!response.ok) throw Error();
      file = await response.blob();
    }
    if (current === revision) getWorker().postMessage(file);
  } catch {
    if (current === revision) reportError("The photo couldn’t be opened. Choose it again and retry.");
  }
};
$("about").onclick = () => $("details").showModal();
$("close").onclick = () => $("details").close();
