"""Export the original B3 and validate its logits before packaging a lossless download."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import urlretrieve

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FoodClassifier101"))
from model import create_effnetb3_model
from weight_layout import weight_layout

SOURCE = "https://huggingface.co/spaces/Zeyad-Sayed/Food-Classifier-101/resolve/54de17314c0f54b3915d7d37b2ea77f613748921/effnetb3.pth"
CHECKSUM = "802a4ab5ba00e3618c374532d4f8993c16d49bc720c8b74f849d943679de755d"


def export(checkpoint, temporary, onnx_path):
    if not checkpoint:
        checkpoint = temporary / "effnetb3.pth"
        urlretrieve(SOURCE, checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKSUM:
        raise ValueError("Checkpoint differs from the published EfficientNet-B3 weights")
    labels = (ROOT / "FoodClassifier101/class_names.txt").read_text().splitlines()
    assert len(set(labels)) == len(labels) == 101
    model, transform = create_effnetb3_model(len(labels))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    torch.set_num_threads(2)
    raw = temporary / "effnetb3.onnx"
    with torch.inference_mode():
        torch.onnx.export(model, torch.zeros(1, 3, 300, 300), raw, opset_version=17,
                          input_names=["image"], output_names=["logits"], do_constant_folding=True)
        onnx.checker.check_model(str(raw))
        runtime = ort.InferenceSession(str(raw), providers=["CPUExecutionProvider"])
        references = {}
        for photo in sorted((ROOT / "web/examples").glob("*.jpg")):
            with Image.open(photo) as image:
                batch = transform(image.convert("RGB")).unsqueeze(0)
            logits = model(batch).numpy()[0]
            actual = runtime.run(None, {"image": batch.numpy()})[0][0]
            np.testing.assert_allclose(actual, logits, atol=4e-5, rtol=1e-5)
            np.testing.assert_array_equal(actual.argsort()[-5:], logits.argsort()[-5:])
            references[photo.name] = logits.tolist()
            print(photo.name, labels[logits.argmax()], "max logit error", abs(actual-logits).max())

    # Shuffle byte lanes before DEFLATE. Grouping sign/exponent bytes improves
    # compression of float weights. A short tail is stored verbatim.
    original = raw.read_bytes()
    (ROOT / "scripts/weight-layout.json").write_text(json.dumps(weight_layout(original), separators=(",", ":")) + "\n")
    if onnx_path:
        onnx_path.write_bytes(original)
    aligned = len(original) - len(original) % 4
    lanes = b"".join(original[i:aligned:4] for i in range(4)) + original[aligned:]
    packed = gzip.compress(lanes, compresslevel=6, mtime=0)
    digest = hashlib.sha256(original).hexdigest()
    directory = ROOT / "web/models"
    directory.mkdir(exist_ok=True)
    name = f"effnetb3-{digest[:16]}.onnx.shuffle.gz"
    (directory / name).write_bytes(packed)
    metadata = dict(file=name, sha256=digest, bytes=len(original), downloadBytes=len(packed),
                    resize=transform.resize_size[0], crop=transform.crop_size[0],
                    mean=transform.mean, std=transform.std, interpolation=transform.interpolation.value)
    assert metadata["interpolation"] == "bicubic"
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (ROOT / "scripts/reference.json").write_text(json.dumps(references, indent=2) + "\n")
    print(f"{len(original):,} ONNX bytes → {len(packed):,} download bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path, help="Also save a standard uncompressed ONNX file")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        export(args.checkpoint, Path(temp), args.onnx)
