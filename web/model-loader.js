import * as ort from "./runtime/ort.wasm.min.mjs";
import { applyCorrection, expandWeights } from "./model-format.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./runtime/", import.meta.url).href;
const assets = new URL("./models/", import.meta.url);
const cache = globalThis.caches?.open("food101-model-v3").catch(() => null);
const listeners = new Set();
let progress = "Preparing model…";
let settings, previewWeights, runtime, quickSession, exactSession, exactReady;

export function watchProgress(callback) {
  listeners.add(callback);
  callback(progress);
  return () => listeners.delete(callback);
}
function report(message) {
  progress = message;
  for (const callback of listeners) callback(message);
}
async function json(file) {
  const response = await fetch(new URL(file, assets));
  if (!response.ok) throw Error("Couldn’t load model settings.");
  return response.json();
}
export function configuration() {
  settings ??= Promise.all([json("web-v3.json"), json("labels.json")])
    .then(([meta, labels]) => ({ meta, labels }))
    .catch(error => { settings = null; throw error; });
  return settings;
}
async function readAsset(part, onBytes) {
  const url = new URL(part.file, assets).href;
  const store = await cache;
  const cached = await store?.match(url).catch(() => null);
  if (cached) {
    const bytes = await cached.arrayBuffer();
    if (bytes.byteLength === part.size) { onBytes(bytes.byteLength); return bytes; }
    await store.delete(url).catch(() => {});
  }
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw Error("Model download interrupted.");
  const bytes = new Uint8Array(part.size);
  const reader = response.body.getReader();
  let offset = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.length > bytes.length) {
      await reader.cancel();
      throw Error("Incorrect download size.");
    }
    bytes.set(value, offset);
    offset += value.length;
    onBytes(value.length);
  }
  if (offset !== part.size) throw Error("Incomplete model download.");
  if (store) await store.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}
async function decompress(parts, phase) {
  const chunks = new Array(parts.length);
  let next = 0, received = 0, percent = -1;
  const total = parts.reduce((sum, part) => sum + part.size, 0);
  async function download() {
    while (next < parts.length) {
      const i = next++;
      chunks[i] = await readAsset(parts[i], length => {
        received += length;
        const progress = Math.floor(100 * received / total);
        if (phase && progress !== percent) report(phase + " · " + progress + "%");
        percent = progress;
      });
    }
  }
  await Promise.all(Array.from({ length: Math.min(4, parts.length) }, download));
  const stream = new Blob(chunks).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
function weights(meta) {
  previewWeights ??= decompress(meta.preview, "Loading quick model")
    .then(bytes => expandWeights(bytes, meta.tensors))
    .catch(error => { previewWeights = null; throw error; });
  return previewWeights;
}
function prepareRuntime(meta) {
  runtime ??= decompress([meta.runtime]).then(bytes => { ort.env.wasm.wasmBinary = bytes; })
    .catch(error => { runtime = null; throw error; });
  return runtime;
}
async function create(bytes, meta) {
  await prepareRuntime(meta);
  return ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
}
export async function isWarm(meta) {
  if (exactReady) return true;
  const store = await cache;
  if (!store) return false;
  const keys = await store.keys().catch(() => []);
  const saved = new Set(keys.map(key => key.url));
  return meta.correction.every(part => saved.has(new URL(part.file, assets).href));
}
export function quick(meta) {
  quickSession ??= Promise.all([weights(meta), prepareRuntime(meta)])
    .then(([bytes]) => create(bytes, meta))
    .catch(error => { quickSession = null; throw error; });
  return quickSession;
}
export function exact(meta) {
  exactSession ??= Promise.all([weights(meta), decompress(meta.correction, "Loading full precision"), prepareRuntime(meta)])
    .then(async ([bytes, correction]) => {
      const restored = applyCorrection(bytes.slice(), correction);
      const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", restored));
      const hex = Array.from(digest, b => b.toString(16).padStart(2, "0")).join("");
      if (hex !== meta.sha256) {
        // Re-fetch model assets after a corrupted cache entry.
        const store = await cache;
        if (store) await Promise.all([...meta.preview, ...meta.correction].map(p => store.delete(new URL(p.file, assets))));
        previewWeights = null;
        throw Error("Model integrity check failed. Retry the download.");
      }
      const session = await create(restored, meta);
      exactReady = true;
      previewWeights = null;
      return session;
    }).catch(error => { exactSession = null; throw error; });
  return exactSession;
}
export async function releaseQuick() {
  if (exactReady && quickSession) {
    const previous = quickSession;
    quickSession = null;
    await (await previous).release();
  }
}

export async function warm() {
  const { meta } = await configuration();
  await (await isWarm(meta) ? exact(meta) : quick(meta));
}
