"""Lab 3 — Dữ liệu và wHash, phần thực hiện của Trần Ngọc Nhân.

Quy ước: ảnh NumPy là uint8, thứ tự kênh RGB; bit dùng phép >= ngưỡng.
Đường dẫn trong chỉ mục tương đối với thư mục chứa dataset_pairs.json.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import pywt

ROOT = Path(__file__).resolve().parent


def load_image(image):
    """Đọc đường dẫn/PIL/NumPy uint8; sửa hướng EXIF, ghép alpha nền trắng."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return load_image(ImageOps.exif_transpose(source))
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("Ảnh NumPy phải có kiểu uint8 và 2 hoặc 3 chiều.")
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError("Ảnh màu phải có 3 kênh RGB hoặc 4 kênh RGBA.")
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError("Cần đường dẫn, ảnh PIL hoặc mảng NumPy uint8.")
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    return image.convert("RGB")


def preprocess_image(image, image_size=32):
    """Chuyển xám, resize LANCZOS về hình vuông, chuẩn hóa float64 [0, 1]."""
    if not isinstance(image_size, int) or image_size < 2:
        raise ValueError("image_size phải là số nguyên >= 2.")
    gray = load_image(image).convert("L").resize(
        (image_size, image_size), Image.Resampling.LANCZOS
    )
    return np.asarray(gray, dtype=np.float64) / 255.0


def _bits(value):
    """Chuẩn hóa chuỗi bit hoặc mảng nhị phân; từ chối đầu vào không hợp lệ."""
    if isinstance(value, str):
        if not value or set(value) - {"0", "1"}:
            raise ValueError("Chuỗi hash chỉ được chứa 0 và 1, không được rỗng.")
        value = [int(bit) for bit in value]
    array = np.asarray(value)
    if array.ndim not in (1, 2) or not array.size or not np.isin(array, [0, 1]).all():
        raise ValueError("Hash phải là vector/ma trận bit không rỗng.")
    return array.astype(bool).ravel()


def hamming_distance(hash1, hash2):
    """Đếm số bit khác nhau; chỉ so sánh hash có cùng độ dài và cấu hình."""
    first, second = _bits(hash1), _bits(hash2)
    if first.size != second.size:
        raise ValueError("Hai hash phải có cùng số bit.")
    return int(np.count_nonzero(first != second))


def similarity_percentage(hash1, hash2):
    """Tính (1 - Hamming / số bit) * 100; đây không phải xác suất cùng ảnh."""
    return 100.0 * (1.0 - hamming_distance(hash1, hash2) / _bits(hash1).size)


def hash_to_hex(bits):
    """Biểu diễn hex có giữ các số 0 đầu; độ dài hash phải chia hết cho 4."""
    bits = _bits(bits)
    if bits.size % 4:
        raise ValueError("Độ dài hash phải chia hết cho 4 để chuyển hex.")
    binary = "".join(bits.astype(np.uint8).astype(str))
    return f"{int(binary, 2):0{bits.size // 4}x}"


class WaveletHasher:
    """wHash từ LL với mode periodization, không loại thành phần DC.

    LL phải có ít nhất hash_size phần tử mỗi chiều. Nếu LL lớn hơn,
    lấy trung bình từng khối đều nhau để thu về hash_size × hash_size.
    Không nội suy phóng lớn LL và không bỏ góc phải/dưới của LL.
    """

    def __init__(self, hash_size=8, image_size=32, wavelet="haar", level=2,
                 threshold="median"):
        for name, value in (("hash_size", hash_size), ("image_size", image_size)):
            if not isinstance(value, int) or value < 2 or value & (value - 1):
                raise ValueError(f"{name} phải là lũy thừa của 2, >= 2.")
        if not isinstance(level, int) or level < 1:
            raise ValueError("level phải là số nguyên >= 1.")
        if threshold not in ("median", "mean"):
            raise ValueError("threshold phải là 'median' hoặc 'mean'.")
        wave = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(image_size, wave.dec_len)
        if level > max_level:
            raise ValueError(f"{wavelet} với ảnh {image_size}: level tối đa {max_level}.")
        if image_size // (2 ** level) < hash_size:
            raise ValueError("LL nhỏ hơn hash_size; hãy tăng image_size hoặc giảm level.")
        self.hash_size, self.image_size = hash_size, image_size
        self.wavelet, self.level, self.threshold = wavelet, level, threshold

    @property
    def n_bits(self):
        """Số bit đầu ra."""
        return self.hash_size ** 2

    def config(self):
        """Cấu hình phải được lưu cùng hash để nhóm tái lập phép so sánh."""
        return dict(hash_size=self.hash_size, image_size=self.image_size,
                    wavelet=self.wavelet, level=self.level, threshold=self.threshold)

    def decompose(self, image):
        """Trả LL, LH=cH, HL=cV, HH=cD cùng cấp sâu nhất và toàn bộ hệ số.

        cH/cV tuân theo trục của PyWavelets, không suy diễn từ hướng hiển thị.
        """
        gray = preprocess_image(image, self.image_size)
        coeffs = pywt.wavedec2(gray, self.wavelet, mode="periodization", level=self.level)
        ch, cv, cd = coeffs[1]
        return dict(gray=gray, LL=coeffs[0], LH=ch, HL=cv, HH=cd, coeffs=coeffs)

    def analyze(self, image):
        """Trả hệ số, ma trận lượng tử, ngưỡng, bit và hex để trực quan hóa."""
        result = self.decompose(image)
        ll = result["LL"]
        block = ll.shape[0] // self.hash_size
        features = ll.reshape(self.hash_size, block, self.hash_size, block).mean(axis=(1, 3))
        cutoff = float(np.median(features) if self.threshold == "median" else np.mean(features))
        bits = (features >= cutoff).ravel()
        result.update(features=features, cutoff=cutoff, bits=bits, hex=hash_to_hex(bits))
        return result

    def hash(self, image):
        """Sinh vector boolean theo thứ tự từng hàng, dài 64 hoặc 256 bit."""
        return self.analyze(image)["bits"]

    def compare(self, first, second):
        """Đọc hai ảnh, tính khoảng cách và phần trăm bit trùng nhau."""
        a, b = self.hash(first), self.hash(second)
        return dict(distance=hamming_distance(a, b), similarity=similarity_percentage(a, b))


def augment_image(image, seed=42):
    """Sinh biến dạng có tham số rõ ràng; nhiễu dùng bộ sinh ngẫu nhiên cục bộ."""
    image = load_image(image)
    rng = np.random.default_rng(seed)
    for angle in (-10, -5, 5, 10):
        yield f"rotate_{angle:+d}", image.rotate(angle, Image.Resampling.BICUBIC,
                                               fillcolor=(255, 255, 255)), {"angle": angle}
    for factor in (0.8, 1.2):
        size = tuple(max(1, round(v * factor)) for v in image.size)
        yield f"scale_{factor}", image.resize(size, Image.Resampling.LANCZOS), {"factor": factor}
        yield f"brightness_{factor}", ImageEnhance.Brightness(image).enhance(factor), {"factor": factor}
    array = np.asarray(image, dtype=np.float64)
    noise = np.clip(array + rng.normal(0, 10, array.shape), 0, 255).astype(np.uint8)
    yield "gaussian_noise", Image.fromarray(noise), {"sigma": 10, "seed": seed}
    noisy = array.astype(np.uint8)
    mask = rng.random(array.shape[:2])
    noisy[mask < 0.01] = 0
    noisy[(mask >= 0.01) & (mask < 0.02)] = 255
    yield "salt_pepper", Image.fromarray(noisy), {"amount": 0.02, "seed": seed}
    for quality in (30, 50, 70):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            yield f"jpeg_{quality}", compressed.convert("RGB"), {"quality": quality}
    yield "blur", image.filter(ImageFilter.GaussianBlur(1.5)), {"radius": 1.5}
    w, h = image.size
    crop = image.crop((round(w * .05), round(h * .05), round(w * .95), round(h * .95)))
    yield "crop", crop.resize(image.size, Image.Resampling.LANCZOS), {"margin": 0.05}


def prepare_originals(data_dir):
    """Thu thập 10 ảnh có sẵn trong scikit-image và vẽ 5 ảnh chữ số bổ sung."""
    from skimage import data

    target = Path(data_dir) / "original"
    target.mkdir(parents=True, exist_ok=True)
    samples = [("astronaut", "chan_dung"), ("camera", "chan_dung"),
               ("chelsea", "dong_vat"), ("coffee", "do_vat"), ("coins", "do_vat"),
               ("clock", "do_vat"), ("rocket", "phong_canh"),
               ("horse", "dong_vat"), ("brick", "ket_cau"), ("grass", "ket_cau")]
    originals = []
    for name, category in samples:
        array = getattr(data, name)()
        if array.dtype == bool:
            array = array.astype(np.uint8) * 255
        image = load_image(array)
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        path = target / f"{name}.png"
        image.save(path)
        originals.append(dict(id=name, category=category, path=f"original/{name}.png",
                              source=f"skimage.data.{name}",
                              source_url=f"https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.{name}",
                              synthetic=False))
    # Chữ số bảy đoạn tự vẽ: không giả định là ảnh chụp hay dữ liệu viết tay.
    segments = [(65, 35, 185, 50), (180, 45, 195, 125), (180, 130, 195, 210),
                (65, 205, 185, 220), (55, 130, 70, 210), (55, 45, 70, 125),
                (65, 120, 185, 135)]
    for digit, indices in {0: [0, 1, 2, 3, 4, 5], 1: [1, 2], 2: [0, 1, 6, 4, 3],
                           3: [0, 1, 6, 2, 3], 4: [5, 6, 1, 2]}.items():
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        for index in indices:
            draw.rectangle(segments[index], fill="black")
        name = f"digit_{digit}"
        image.save(target / f"{name}.png")
        originals.append(dict(id=name, category="chu_so", path=f"original/{name}.png",
                              source="Tự vẽ chữ số bảy đoạn bằng Pillow", synthetic=True))
    return originals


def build_dataset(data_dir=ROOT / "data", seed=42):
    """Tạo 225 cặp dương và 105 cặp âm duy nhất; lưu nguồn và nhóm ảnh gốc.

    Ảnh JPEG được giải mã rồi lưu PNG để giữ đúng một lần nén. Cặp âm tham
    chiếu trực tiếp hai ảnh gốc khác nhau, không sao chép thư mục dissimilar.
    """
    data_dir = Path(data_dir)
    originals = prepare_originals(data_dir)
    augmented = data_dir / "augmented_similar"
    augmented.mkdir(parents=True, exist_ok=True)
    pairs = []
    for index, original in enumerate(originals):
        for kind, image, parameters in augment_image(data_dir / original["path"], seed + index):
            relative = f"augmented_similar/{original['id']}__{kind}.png"
            image.save(data_dir / relative)
            pairs.append(dict(image1=original["path"], image2=relative, label=1,
                              source_id1=original["id"], source_id2=original["id"],
                              transform=kind, parameters=parameters))
    for first, second in itertools.combinations(originals, 2):
        pairs.append(dict(image1=first["path"], image2=second["path"], label=0,
                          source_id1=first["id"], source_id2=second["id"],
                          transform="different_source", parameters={}))
    for index, pair in enumerate(pairs):
        pair["id"] = f"pair_{index:04d}"
    manifest = dict(schema_version=1, seed=seed, originals=originals, pairs=pairs,
                    label_definition={"1": "Cùng ảnh nguồn và biến dạng", "0": "Khác ảnh nguồn"})
    (data_dir / "dataset_pairs.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def compute_pair_distances(data_dir=ROOT / "data", hasher=None):
    """Xuất khoảng cách để bàn giao Phase 3; mỗi ảnh chỉ băm một lần."""
    data_dir, hasher = Path(data_dir), hasher or WaveletHasher()
    manifest = json.loads((data_dir / "dataset_pairs.json").read_text(encoding="utf-8"))
    paths = sorted({pair[key] for pair in manifest["pairs"] for key in ("image1", "image2")})
    hashes = {path: hasher.hash(data_dir / path) for path in paths}
    rows = []
    for pair in manifest["pairs"]:
        a, b = hashes[pair["image1"]], hashes[pair["image2"]]
        rows.append({**pair, "distance": hamming_distance(a, b),
                     "similarity": similarity_percentage(a, b)})
    return dict(config=hasher.config(), n_bits=hasher.n_bits,
                hashes={path: hash_to_hex(bits) for path, bits in hashes.items()}, pairs=rows)


def main():
    """Tạo dữ liệu khi chưa có chỉ mục và xuất kết quả bàn giao có thể tái lập."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--rebuild", action="store_true", help="Tái tạo các ảnh mẫu do chương trình quản lý")
    args = parser.parse_args()
    if args.rebuild or not (args.data_dir / "dataset_pairs.json").exists():
        build_dataset(args.data_dir)
    result = compute_pair_distances(args.data_dir)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "pair_distances.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(result['hashes'])} images; {len(result['pairs'])} pairs; {result['n_bits']}-bit wHash")
    print(output / "pair_distances.json")


if __name__ == "__main__":
    main()
