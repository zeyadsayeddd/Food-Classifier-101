// Pillow's antialiased bicubic filter, with its per-axis 8-bit rounding.
// Evaluate only pixels used by the subsequent torchvision center crop.
function cubic(x) {
  x = Math.abs(x);
  if (x < 1) return (1.5 * x - 2.5) * x * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}
function roundEven(x) {
  const floor = Math.floor(x);
  return x - floor === 0.5 ? floor + floor % 2 : Math.round(x);
}
function filters(input, output, crop) {
  const ratio = input / output;
  const scale = Math.max(1, ratio);
  const offset = roundEven((output - crop) / 2);
  return Array.from({ length: crop }, (_, i) => {
    const center = (i + offset + 0.5) * ratio;
    const start = Math.max(0, Math.floor(center - 2 * scale + 0.5));
    const end = Math.min(input, Math.floor(center + 2 * scale + 0.5));
    const weights = Array.from({ length: end - start }, (_, k) => cubic((start + k + 0.5 - center) / scale));
    const sum = weights.reduce((a, b) => a + b, 0);
    return { start, weights: weights.map(w => Math.sign(w) * Math.floor(Math.abs(w / sum) * 4194304 + 0.5)) };
  });
}
const pixel = sum => Math.max(0, Math.min(255, Math.floor(sum / 4194304 + 0.5)));
export function preprocess({ data, width, height }, { resize, crop, mean, std }) {
  const short = Math.min(width, height);
  const xs = filters(width, Math.floor(width * resize / short), crop);
  const ys = filters(height, Math.floor(height * resize / short), crop);
  const first = ys[0].start;
  const last = ys[crop - 1];
  const rows = last.start + last.weights.length - first;
  const temp = new Uint8Array(rows * crop * 3);
  for (let row = 0; row < rows; row++) {
    for (let x = 0; x < crop; x++) {
      const { start, weights } = xs[x];
      const src = ((first + row) * width + start) * 4;
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let k = 0; k < weights.length; k++) sum += data[src + 4 * k + c] * weights[k];
        temp[(row * crop + x) * 3 + c] = pixel(sum);
      }
    }
  }
  const plane = crop * crop;
  const output = new Float32Array(plane * 3);
  for (let y = 0; y < crop; y++) {
    const { start, weights } = ys[y];
    for (let x = 0; x < crop; x++) {
      const src = ((start - first) * crop + x) * 3;
      for (let c = 0; c < 3; c++) {
        let sum = 0;
        for (let k = 0; k < weights.length; k++) sum += temp[src + 3 * crop * k + c] * weights[k];
        const centered = Math.fround(Math.fround(pixel(sum) / 255) - Math.fround(mean[c]));
        output[c * plane + y * crop + x] = centered / Math.fround(std[c]);
      }
    }
  }
  return output;
}
