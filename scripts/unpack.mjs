import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";

export async function originalModel() {
  const dir = new URL("../web/models/", import.meta.url);
  const meta = JSON.parse(await readFile(new URL("manifest.json", dir), "utf8"));
  const packed = await readFile(new URL(meta.file, dir));
  const lanes = gunzipSync(packed);
  const model = Buffer.from(lanes);
  const n = Math.floor(model.length / 4);
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < n; i++) model[4 * i + j] = lanes[j * n + i];
  }
  if (model.length !== meta.bytes || createHash("sha256").update(model).digest("hex") !== meta.sha256) {
    throw Error("ONNX integrity check failed");
  }
  return model;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const model = await originalModel();
  if (process.argv[2]) await writeFile(process.argv[2], model);
  console.log("Verified EfficientNet-B3 ONNX");
}
