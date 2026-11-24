import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import { originalModel } from "./unpack.mjs";
import { expandWeights, applyCorrection } from "../web/model-format.js";

export const hash = data => createHash("sha256").update(data).digest("hex");

async function asset(name, data) {
  const file = name + "-" + hash(data).slice(0, 16) + ".gz";
  await writeFile("dist/models/" + file, data);
  return { file, size: data.length };
}

async function chunks(name, bytes) {
  const packed = gzipSync(bytes, { level: 6 });
  const parts = [];
  for (let from = 0; from < packed.length; from += 2 ** 20) {
    parts.push(await asset(name, packed.subarray(from, from + 2 ** 20)));
  }
  return parts;
}

export async function packWeb(meta) {
  const original = await originalModel();
  const layout = JSON.parse(await readFile("scripts/weight-layout.json", "utf8"));
  if (layout.sha256 !== meta.sha256) throw Error("Weight layout differs from model");
  const packed = Buffer.from(original);
  for (const [start, rows, columns] of layout.tensors) {
    const count = rows * columns;
    packed.fill(0, start, start + count * 4);
    for (let row = 0; row < rows; row++) {
      const offset = row * columns;
      let max = 0;
      for (let col = 0; col < columns; col++) {
        max = Math.max(max, Math.abs(original.readFloatLE(start + 4 * (offset + col))));
      }
      const scale = Math.fround(max / 127) || 1;
      packed.writeFloatLE(scale, start + row * 4);
      for (let col = 0; col < columns; col++) {
        const i = offset + col;
        const value = original.readFloatLE(start + 4 * i);
        packed.writeInt8(Math.max(-127, Math.min(127, Math.round(value / scale))), start + 4 * rows + i);
      }
    }
  }
  const preview = await chunks("preview", packed);
  const approx = expandWeights(packed, layout.tensors);
  const lanes = Buffer.alloc(original.length);
  const n = Math.floor(original.length / 4);
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < n; i++) lanes[j * n + i] = original[i * 4 + j] ^ approx[i * 4 + j];
  }
  for (let i = n * 4; i < original.length; i++) lanes[i] = original[i] ^ approx[i];
  const correction = await chunks("precision", lanes);
  if (hash(applyCorrection(approx, lanes)) !== meta.sha256) throw Error("Exact reconstruction failed");
  const wasm = gzipSync(await readFile("node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"), { level: 9 });
  const runtime = await asset("wasm", wasm);
  const { file, downloadBytes, ...settings } = meta;
  const manifest = { ...settings, preview, correction, runtime, tensors: layout.tensors };
  await writeFile("dist/models/web-v3.json", JSON.stringify(manifest));
  const size = parts => parts.reduce((sum, part) => sum + part.size, 0);
  console.log("Download bytes", { preview: size(preview), correction: size(correction), runtime: runtime.size });
}
