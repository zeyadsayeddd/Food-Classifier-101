import { packWeb, hash } from "./pack-web.mjs";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";

const labels = (await readFile("FoodClassifier101/class_names.txt", "utf8")).trim().split(/\r?\n/);
if (labels.length !== 101 || new Set(labels).size !== 101) throw Error("Expected 101 ordered class names");
const meta = JSON.parse(await readFile("web/models/manifest.json", "utf8"));
await rm("dist", { recursive: true, force: true });
await mkdir("dist/runtime", { recursive: true });
await cp("web", "dist", { recursive: true, filter: p => !p.includes(".onnx.") && !p.endsWith("models/manifest.json") && !p.endsWith("sample-results.json") });
await writeFile("dist/models/labels.json", JSON.stringify(labels));
for (const asset of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs"]) {
  await cp("node_modules/onnxruntime-web/dist/" + asset, "dist/runtime/" + asset);
}
await packWeb(meta);
let examples = {};
if (!process.argv.includes("--for-sample-capture")) {
  const saved = JSON.parse(await readFile("web/sample-results.json", "utf8"));
  if (saved.model !== meta.sha256) throw Error("Refresh cached sample results for this model");
  for (const [name, photo] of Object.entries(saved.photos)) {
    if (hash(await readFile("web/examples/" + name)) !== photo.sha256) throw Error("Refresh cached results for " + name);
  }
  examples = saved.photos;
}
await writeFile("dist/samples-v3.js", "export const examples = " + JSON.stringify(examples));
await mkdir("dist/fonts");
for (const [pkg, font] of [
  ["@fontsource-variable/dm-sans", "dm-sans-latin-wght-normal"],
  ["@fontsource/instrument-serif", "instrument-serif-latin-400-normal"],
]) {
  await cp("node_modules/" + pkg + "/files/" + font + ".woff2", "dist/fonts/" + font + ".woff2");
  await cp("node_modules/" + pkg + "/LICENSE", "dist/fonts/" + font + "-LICENSE.txt");
}
