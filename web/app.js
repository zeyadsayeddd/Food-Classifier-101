const $ = id => document.getElementById(id);
const analyze = $("analyze");
const imageName = $("image-name");
let selected = { src: $("preview").src };
let worker, objectUrl, revision = 0, busy = false;
let queue = Promise.resolve();

function showProgress(message) {
  analyze.querySelector("span").textContent = message;
  if (selected.button) $("source").textContent = message;
}
function ready() {
  busy = false;
  analyze.disabled = false;
  analyze.querySelector("span").textContent = "Identify dish";
}
function select(photo) {
  revision++;
  $("announcement").textContent = "";
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
const samples = [...$("samples").children];
function trySample(button) {
  select({ button, name: button.dataset.name, src: button.querySelector("img").src });
  identify();
}
for (const button of samples) button.onclick = () => trySample(button);
$("try-sample").onclick = () => {
  const current = samples.findIndex(button => button.getAttribute("aria-pressed") === "true");
  trySample(samples[(current + 1) % samples.length]);
};
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
  const summary = "Best match: " + rows[0].querySelector("strong").textContent;
  $("announcement").textContent = summary;
  if (selected.button) $("source").textContent = summary;
}
function reportError(message) {
  ready();
  if (selected.button) $("source").textContent = "Sample photo";
  $("notice").textContent = message;
}
function runPrediction(file, current) {
  if (current !== revision) return;
  worker ??= new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      if (data.status) {
        if (current === revision || (busy && data.status.startsWith("Loading model"))) {
          showProgress(data.status);
        }
        return;
      }
      if (current === revision) {
        if (data.error) reportError(data.error);
        else { ready(); showMatches(data); }
      }
      resolve();
    };
    worker.onerror = () => {
      worker.terminate();
      worker = null;
      reject(new Error("Couldn’t start the classifier. Check your connection and retry."));
    };
    worker.postMessage(file);
  });
}
async function identify() {
  if (busy) return;
  const current = revision;
  busy = true;
  analyze.disabled = true;
  showProgress("Preparing photo…");
  $("notice").textContent = "";
  try {
    let file = selected.file;
    if (!file) {
      const response = await fetch(selected.src);
      if (!response.ok) throw Error();
      file = await response.blob();
    }
    if (current !== revision) return;
    showProgress("Preparing model…");
    // Finish any current inference and reuse its model download for the latest selection.
    queue = queue.then(() => runPrediction(file, current)).catch(error => {
      if (current === revision) reportError(error.message);
    });
  } catch {
    if (current === revision) reportError("The photo couldn’t be opened. Choose it again and retry.");
  }
}
analyze.onclick = identify;
$("about").onclick = () => $("details").showModal();
$("close").onclick = () => $("details").close();
