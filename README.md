# Food Classifier 101

[Open Dish Index on GitHub Pages](https://zeyadsayeddd.github.io/Food-Classifier-101/). Click **Try a sample** for its saved top five. These results were computed in Chromium with the original model. **Run this sample on my device** repeats the inference live. For your own photo, drop the image and select **Identify dish**. Inference runs in a browser worker using ONNX Runtime Web and WebAssembly. Photos stay on the device.

## Model and preprocessing

This project originally compared EfficientNet-B2 and B3 on a 20% subset of Food-101 and selected B3. The browser version uses the original 101-output EfficientNet-B3 checkpoint. The quick estimate uses 8-bit weights per output channel. Full-precision predictions use the byte-identical original float32 ONNX graph. Both run without retraining. The [training notebook](food-classifier-101.ipynb), [Kaggle notebook](https://www.kaggle.com/code/zeyadsayedadbullah/food-classifier-101/), and [original checkpoint and sample photos](https://huggingface.co/spaces/Zeyad-Sayed/Food-Classifier-101/tree/54de17314c0f54b3915d7d37b2ea77f613748921) document its provenance.

The browser decodes the photo to RGB, resizes its shorter edge to 320 with antialiased bicubic interpolation, and takes a 300 × 300 center crop using torchvision rounding. It scales channels to [0, 1], normalizes by ImageNet mean and standard deviation, and packs a float32 NCHW tensor. The model emits logits. Softmax scores use the [original ordered class names](FoodClassifier101/class_names.txt).

## Browser loading and speed

For uploads, the page starts preparing the model when you select the photo. It fetches the WebAssembly runtime and compact weights concurrently, computes an initial **Quick estimate**, then automatically applies the precision data and updates the result with full-precision scores. The build and browser verify the restored graph against the original SHA-256. The compact model can rank nearby labels differently.

| Cold download | Bytes |
| --- | ---: |
| Runtime, compressed | 2,892,500 |
| Weights needed for the quick estimate | 10,220,384 |
| Additional data to restore the original model | 29,604,938 |

The app requests up to four weight chunks at a time. The Cache API stores each completed chunk under a content-hashed URL for later photos and visits. If browser storage is unavailable, the page falls back to HTTP caching. Fixed samples need only their saved scores and the normal page assets. The source archive still contains the original losslessly packed 37,164,257-byte model for standalone ONNX export.

Measured in local Chromium with model and runtime traffic sharing a 10 Mbps limit. Each cold result starts at the corresponding user action. The reload comparison uses the same browser context and includes a fresh inference session.

| Action | Previous | Updated |
| --- | ---: | ---: |
| Open a built-in sample on a fresh visit | 38.12 s | 0.08 s, saved result |
| First uploaded photo, first result | 37.68 s | 11.41 s, quick estimate |
| First uploaded photo, full precision | 37.68 s | 35.32 s |
| Uploaded photo after reload, full precision | 35.01 s | 0.84 s |

On 92 additional Food-101 validation photos spanning 46 classes, the preview and original agreed on 84 top labels. These figures describe the preview on that subset. Exact results automatically replace previews. Actual load times depend on the device, available storage and network.

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

If a changed model or sample image invalidates the saved examples, build with `npm run build -- --for-sample-capture`, serve `dist`, run `python scripts/cache_samples.py http://localhost:8000/` with Playwright installed, and rebuild normally.

To use a local copy of the original weights, pass `--checkpoint /path/to/effnetb3.pth`. To repeat the browser comparison with the local server running, install Playwright and Chromium in the same environment.

```sh
pip install playwright
python -m playwright install chromium
python scripts/verify_browser.py http://localhost:8000/
```

On the three supplied photos (pizza, steak, and sushi), the native ONNX and Chromium runs reproduced PyTorch’s top-five order. The maximum native ONNX logit difference was 0.0000182. The largest browser softmax probability difference was below 0.00000053 across the 303 scores. The parity test exercises saved samples, live estimates, full-precision predictions, uploads, and a reload with model downloads blocked.

Original model and training by Zeyad Sayed.
