"""Save full-precision ONNX Runtime Web predictions for the three fixed demo files."""
import argparse
import hashlib
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("site", help="URL of the locally served production build")
args = parser.parse_args()
cache = {"model": json.loads((root / "web/models/manifest.json").read_text())["sha256"], "photos": {}}
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(args.site.rstrip("/") + "/")
    served = page.evaluate("async () => (await (await fetch('models/web-v3.json')).json()).sha256")
    if served != cache["model"]:
        raise ValueError("Served model differs from the export. Rebuild before capturing.")
    for photo in sorted((root / "web/examples").glob("*.jpg")):
        matches = page.evaluate("""async name => {
          const {predict} = await import('./inference.js');
          const blob = await (await fetch('examples/' + name)).blob();
          return (await predict(blob)).matches;
        }""", photo.name)
        cache["photos"][photo.name] = {"sha256": hashlib.sha256(photo.read_bytes()).hexdigest(), "matches": matches}
    browser.close()
(root / "web/sample-results.json").write_text(json.dumps(cache, indent=2) + "\n")
