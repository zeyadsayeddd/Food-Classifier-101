# Food Classifier 101

[Open Dish Index on GitHub Pages](https://zeyadsayeddd.github.io/Food-Classifier-101/). Choose a sample or drop a food photo, then select **Identify dish** for the five highest-scoring Food-101 labels. Inference runs in a browser worker using ONNX Runtime Web and WebAssembly. Photos stay on the device.

## Model and preprocessing

This project originally compared EfficientNet-B2 and B3 on a 20% subset of Food-101 and selected B3. The browser version uses the original 101-output EfficientNet-B3 checkpoint. No retraining or quantization is involved. The [training notebook](food-classifier-101.ipynb), [Kaggle notebook](https://www.kaggle.com/code/zeyadsayedadbullah/food-classifier-101/), and [original checkpoint and sample photos](https://huggingface.co/spaces/Zeyad-Sayed/Food-Classifier-101/tree/54de17314c0f54b3915d7d37b2ea77f613748921) document its provenance.

The browser decodes the photo to RGB, resizes its shorter edge to 320 with antialiased bicubic interpolation, and takes a 300 × 300 center crop using torchvision rounding. It scales channels to [0, 1], normalizes by ImageNet mean and standard deviation, and packs a float32 NCHW tensor. The model emits logits. Softmax scores use the [original ordered class names](FoodClassifier101/class_names.txt).

| Artifact | Bytes |
| --- | ---: |
| Original PyTorch checkpoint | 43,961,400 |
| Standard float32 ONNX, opset 17 | 43,351,251 |
| Losslessly packed model download | 37,164,257 |

The page loads the WASM-only runtime and packed model on the first analysis. A content-addressed URL lets the browser reuse the downloaded model on later visits. The worker reuses its inference session between photos. Runtime files, fonts, and samples are served from the same GitHub Pages site.

## Use the ONNX file

With Node.js 22 or later, extract the standard model from the included download. The script checks the restored graph’s SHA-256 and needs no npm packages.

```sh
node scripts/unpack.mjs effnetb3-food101.onnx
```

Input is `image` with shape `[1, 3, 300, 300]`. Output is `logits` with shape `[1, 101]`.

## Run locally and publish

```sh
npm ci --ignore-scripts
npm run build
python3 -m http.server 8000 --directory dist
```

Open [localhost:8000](http://localhost:8000). The [Pages workflow](.github/workflows/pages.yml) verifies the packed graph, builds the static files and deploys on pushes to main. The runtime uses module workers, OffscreenCanvas, WebAssembly SIMD, and DecompressionStream in modern browsers.

## Reproduce the conversion and checks

In a Python 3.10 virtual environment, run the following commands. The exporter downloads the pinned checkpoint, verifies its SHA-256, exports in evaluation mode, and checks all 101 logits and the top-five order against PyTorch for every bundled example.

```sh
pip install -r FoodClassifier101/requirements.txt
python scripts/export.py --onnx effnetb3-food101.onnx
npm run build
```

To use a local copy of the original weights, pass `--checkpoint /path/to/effnetb3.pth`. To repeat the browser comparison with the local server running, install Playwright and Chromium in the same environment.

```sh
pip install playwright
python -m playwright install chromium
python scripts/verify_browser.py http://localhost:8000/
```

On the three supplied photos (pizza, steak, and sushi), the native ONNX and Chromium runs reproduced PyTorch’s top-five order. The maximum native ONNX logit difference was 0.0000182. The largest browser softmax probability difference was below 0.00000053 across the 303 scores. These checks cover the bundled examples. A full Food-101 test-set evaluation was outside this comparison.

Original model and training by Zeyad Sayed.
