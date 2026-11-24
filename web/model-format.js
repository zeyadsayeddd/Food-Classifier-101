// Each weight matrix stores one float32 scale per output channel, then signed
// 8-bit coefficients. Other graph bytes, including biases, retain their precision.
export function expandWeights(bytes, tensors) {
  const floats = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (const [start, rows, columns] of tensors) {
    const coefficients = start + rows * 4;
    const scales = Array.from({ length: rows }, (_, i) => floats.getFloat32(start + 4 * i, true));
    const packed = new Int8Array(bytes.buffer, bytes.byteOffset + coefficients, rows * columns).slice();
    for (let row = 0; row < rows; row++) {
      const scale = scales[row];
      for (let col = 0; col < columns; col++) {
        const index = row * columns + col;
        floats.setFloat32(start + 4 * index, packed[index] * scale, true);
      }
    }
  }
  return bytes;
}

export function applyCorrection(weights, lanes) {
  if (weights.length !== lanes.length) throw Error("Incomplete precision data");
  const n = Math.floor(weights.length / 4);
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < n; i++) weights[4 * i + j] ^= lanes[j * n + i];
  }
  for (let i = 4 * n; i < weights.length; i++) weights[i] ^= lanes[i];
  return weights;
}
