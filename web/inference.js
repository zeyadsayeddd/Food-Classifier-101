import * as ort from "./runtime/ort.wasm.min.mjs";
import { preprocess } from "./preprocess.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./runtime/", import.meta.url).href;

async function get(path, options) {
  const response = await fetch(new URL(path, import.meta.url), options);
  if (!response.ok) throw Error("Could not load " + path);
  return response;
}
async function modelBytes(meta, status) {
  const response = await get("./models/" + meta.file, { cache: "force-cache" });
  let received = 0;
  let previous = -1;
  const progress = new TransformStream({
    transform(chunk, controller) {
      received += chunk.byteLength;
      const percent = Math.min(100, Math.floor(received / meta.downloadBytes * 100));
      if (percent !== previous) status("Loading model · " + percent + "%");
      previous = percent;
      controller.enqueue(chunk);
    }
  });
  const input = new Uint8Array(await new Response(response.body.pipeThrough(progress).pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  if (input.length !== meta.bytes) throw Error("Incomplete model");
  const n = Math.floor(input.length / 4);
  const bytes = new Uint8Array(input.length);
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < n; i++) bytes[i * 4 + j] = input[j * n + i];
  }
  bytes.set(input.subarray(n * 4), n * 4);
  return bytes;
}
let resources;
async function load(status) {
  const [meta, labels] = await Promise.all([
    get("./models/manifest.json").then(r => r.json()), get("./models/labels.json").then(r => r.json()),
  ]);
  const session = await ort.InferenceSession.create(await modelBytes(meta, status), { executionProviders: ["wasm"] });
  return { meta, labels, session };
}
export async function predict(file, status = () => {}) {
  status("Preparing model…");
  resources ??= load(status).catch(error => { resources = null; throw error; });
  const { meta, labels, session } = await resources;
  status("Analyzing photo…");
  const start = performance.now();
  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  try {
    ctx.drawImage(bitmap, 0, 0);
  } finally { bitmap.close(); }
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const input = new ort.Tensor("float32", preprocess(rgba, meta), [1, 3, meta.crop, meta.crop]);
  const output = await session.run({ image: input });
  const logits = Array.from(output.logits.data);
  input.dispose();
  output.logits.dispose();
  const max = Math.max(...logits);
  const exp = logits.map(x => Math.exp(x - max));
  const total = exp.reduce((a, b) => a + b, 0);
  const matches = labels.map((label, i) => ({ label, probability: exp[i] / total }))
    .sort((a, b) => b.probability - a.probability).slice(0, 5);
  return { logits, matches, seconds: (performance.now() - start) / 1000 };
}
