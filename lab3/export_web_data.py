"""Export sample images and precomputed wHash from data/ to web/js/dataset_store.js.

Generates base64 thumbnails and reference wHash for both 64-bit (8x8) and 256-bit (16x16).
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
from PIL import Image

from vision_wavelet import WaveletHasher, hash_to_hex

ROOT = Path(__file__).resolve().parent
ORIGINAL_DIR = ROOT / "data" / "original"
AUG_DIR = ROOT / "data" / "augmented_similar"
OUTPUT_FILE = ROOT / "web" / "js" / "dataset_store.js"

hasher_64 = WaveletHasher(hash_size=8, wavelet="haar", level=1, threshold="median")
hasher_256 = WaveletHasher(hash_size=16, wavelet="haar", level=1, threshold="median")


def image_to_base64(path: Path, max_size: int = 140) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = []

    # 1. 15 Original Images
    for path in sorted(ORIGINAL_DIR.glob("*.png")):
        b64 = image_to_base64(path)
        h64 = hasher_64.hash(path)
        h256 = hasher_256.hash(path)
        records.append({
            "id": path.stem,
            "name": path.stem.replace("_", " ").title(),
            "category": "digits" if "digit" in path.stem else "photos",
            "is_original": True,
            "source": path.name,
            "data_url": b64,
            "whash_64_hex": hash_to_hex(h64),
            "whash_64_bits": h64.astype(int).tolist(),
            "whash_256_hex": hash_to_hex(h256),
            "whash_256_bits": h256.astype(int).tolist(),
        })

    # 2. Select representative augmented variations
    key_augs = [
        "astronaut__blur", "astronaut__gaussian_noise", "astronaut__rotate_+10", "astronaut__jpeg_30",
        "camera__brightness_1.2", "camera__rotate_-10", "camera__jpeg_30",
        "chelsea__salt_pepper", "chelsea__blur", "chelsea__scale_0.8",
        "coffee__rotate_+5", "coffee__brightness_0.8",
        "digit_0__rotate_+10", "digit_2__blur", "digit_3__gaussian_noise",
        "rocket__brightness_1.2", "horse__scale_1.2", "coins__gaussian_noise"
    ]
    for stem in key_augs:
        path = AUG_DIR / f"{stem}.png"
        if path.exists():
            b64 = image_to_base64(path)
            h64 = hasher_64.hash(path)
            h256 = hasher_256.hash(path)
            source_stem = stem.split("_")[0]
            records.append({
                "id": stem,
                "name": stem.replace("_", " ").title(),
                "category": "augmented",
                "is_original": False,
                "source": f"{source_stem}.png",
                "data_url": b64,
                "whash_64_hex": hash_to_hex(h64),
                "whash_64_bits": h64.astype(int).tolist(),
                "whash_256_hex": hash_to_hex(h256),
                "whash_256_bits": h256.astype(int).tolist(),
            })

    js_content = (
        "// Auto-generated dataset store from Lab 3 data/ for client-side Web Studio\n"
        f"const SAMPLE_DATASET = {json.dumps(records, indent=2)};\n\n"
        "if (typeof module !== 'undefined' && module.exports) {\n"
        "  module.exports = { SAMPLE_DATASET };\n"
        "}\n"
    )
    OUTPUT_FILE.write_text(js_content, encoding="utf-8")
    print(f"Exported {len(records)} images to {OUTPUT_FILE} (size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
