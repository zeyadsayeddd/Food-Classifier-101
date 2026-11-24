"""Run against a built site. Requires playwright, Chromium, Pillow, and NumPy."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("url")
args = parser.parse_args()
site = args.url.rstrip("/") + "/"
reference = json.loads((ROOT / "scripts/reference.json").read_text())
labels = (ROOT / "FoodClassifier101/class_names.txt").read_text().splitlines()


def probabilities(logits):
    exp = np.exp(np.asarray(logits) - max(logits))
    return exp / exp.sum()


with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 840})
    requests = []
    page.on("request", lambda r: requests.append((r.method, r.url)))
    page.on("pageerror", lambda e: print("Page error:", e, flush=True))
    page.on("console", lambda m: print("Browser:", m.text[:400], flush=True) if m.type == "error" else None)
    page.goto(site)
    assert not any("onnx.shuffle" in url or url.endswith(".wasm") for _, url in requests)
    page.get_by_role("button", name="About the model").click()
    page.get_by_role("button", name="Close").click()
    page.get_by_role("button", name="Identify dish").click()
    page.locator("#ranking li").first.wait_for(timeout=180000)
    for photo, expected in reference.items():
        # Run the production pipeline on the browser's decode, then on Pillow's
        # RGB pixels as PNG to isolate browser JPEG decoder differences.
        with Image.open(ROOT / "web/examples" / photo) as im:
            png = BytesIO()
            im.convert("RGB").save(png, format="PNG")
        for kind, pixels in (("JPEG", None), ("RGB", list(png.getvalue()))):
            result = page.evaluate("""async ([url, pixels]) => {
              const {predict} = await import('./inference.js');
              const file = pixels ? new Blob([new Uint8Array(pixels)], {type:'image/png'}) : await (await fetch(url)).blob();
              return await predict(file);
            }""", [urljoin(site, "examples/" + photo), pixels])
            actual = np.asarray(result["logits"])
            np.testing.assert_array_equal(actual.argsort()[-5:], np.argsort(expected)[-5:])
            difference = np.max(abs(probabilities(actual) - probabilities(expected)))
            assert difference < (0.005 if kind == "JPEG" else 0.00005), (photo, kind, difference)
            print(photo, kind, result["matches"][0]["label"], "max probability delta", difference, flush=True)
    page.set_viewport_size({"width": 375, "height": 812})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.get_by_role("button", name="Use sushi sample").click()
    page.locator("#file").set_input_files(str(ROOT / "web/examples/3622237.jpg"))
    page.get_by_role("button", name="Identify dish").click()
    page.locator("#ranking li").first.wait_for(timeout=180000)
    assert page.locator("#ranking li strong").first.inner_text() == "Steak"
    assert all(method == "GET" and urlparse(url).netloc == urlparse(site).netloc
               for method, url in requests if url.startswith("http"))
    browser.close()
    print("Upload, sample selection, model details, mobile layout, and same-origin GET-only requests passed.")
