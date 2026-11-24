import * as ort from "./runtime/ort.wasm.min.mjs";
import { preprocess } from "./preprocess.js";
import { configuration, exact, isWarm, quick, releaseQuick, watchProgress } from "./model-loader.js";

async function imageTensor(file, meta) {
  const bitmap = await createImageBitmap(file);
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  try {
    ctx.drawImage(bitmap, 0, 0);
  } finally { bitmap.close(); }
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return new ort.Tensor("float32", preprocess(rgba, meta), [1, 3, meta.crop, meta.crop]);
}

async function classify(session, input, labels) {
  const start = performance.now();
  const output = await session.run({ image: input });
  const logits = Array.from(output.logits.data);
  output.logits.dispose();
  const max = Math.max(...logits);
  const exp = logits.map(x => Math.exp(x - max));
  const total = exp.reduce((a, b) => a + b, 0);
  const matches = labels.map((label, i) => ({ label, probability: exp[i] / total }))
    .sort((a, b) => b.probability - a.probability).slice(0, 5);
  return { logits, matches, seconds: (performance.now() - start) / 1000 };
}

// Serialize only execution. New photos can use the quick model while the
// precision data for a previous selection is downloading.
let executions = Promise.resolve();
function run(task) {
  const result = executions.then(task);
  executions = result.catch(() => {});
  return result;
}
let active = 0;
export async function predict(file, status = () => {}, preview, current = () => true) {
  active++;
  const stop = watchProgress(status);
  let tensor;
  try {
    const { meta, labels } = await configuration();
    const staged = preview && !await isWarm(meta);
    const first = staged ? quick(meta) : exact(meta);
    // Wait for both so a decoded tensor is disposed even if session setup fails.
    const [input, ready] = await Promise.allSettled([imageTensor(file, meta), first]);
    if (input.status === "rejected") throw input.reason;
    tensor = input.value;
    if (ready.status === "rejected") throw ready.reason;
    const session = ready.value;
    if (!current()) return;
    if (staged) {
      status("Analyzing photo…");
      const estimate = await run(() => classify(session, tensor, labels));
      if (!current()) return;
      preview(estimate);
      status("Loading full precision…");
    }
    const full = staged ? await exact(meta) : session;
    if (!current()) return;
    status("Computing full-precision scores…");
    const result = await run(() => classify(full, tensor, labels));
    return result;
  } finally {
    tensor?.dispose();
    stop();
    if (--active === 0) await run(releaseQuick);
  }
}
