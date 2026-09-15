"""Tạo notebook tiếng Việt cho Phase 1–2; chạy lại nếu cần sửa cấu trúc báo cáo."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
cells = []


def md(text):
    """Thêm phần giải thích."""
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    """Thêm ô thực nghiệm."""
    cells.append(nbf.v4.new_code_cell(text))


md("""# Lab 3 — Dữ liệu và Wavelet Hashing
**Sinh viên: Trần Ngọc Nhân** · Phụ trách **Phase 1 và Phase 2**.

Notebook trình bày dữ liệu, biến dạng, DWT 2D, wHash 64/256 bit và Hamming.
Đánh giá ROC, chọn ngưỡng và benchmark thuộc phần Thanh Nguyên; Web Studio thuộc Minh Quân.
Kết quả trong notebook được tính từ mã nguồn, không điền số liệu thủ công.
""")
code("""from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt
import pywt
from PIL import Image

# Hỗ trợ mở notebook từ lab3, thư mục dự án hoặc thư mục chứa dự án.
ROOT = next(p.resolve() for p in [Path.cwd(), Path.cwd() / 'lab3',
    Path.cwd() / 'Computer_Vision_Lab-main/lab3'] if (p / 'vision_wavelet.py').exists())
sys.path.insert(0, str(ROOT))
from vision_wavelet import (WaveletHasher, build_dataset, compute_pair_distances,
    hamming_distance, similarity_percentage, hash_to_hex)
DATA, OUT = ROOT / 'data', ROOT / 'results'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
print('NumPy:', np.__version__, '| PyWavelets:', pywt.__version__)
""")
md("""## 1. Chuẩn bị dữ liệu — các mục 1.1–1.5

Dùng 10 ảnh mẫu từ `scikit-image` và 5 chữ số bảy đoạn tự vẽ: tổng cộng 15 ảnh nguồn.
Đây là bộ dữ liệu minh họa nhỏ, có cả ảnh tổng hợp; không đại diện cho ảnh chụp ở góc nhìn khác.
Mỗi nguồn có 15 biến thể: xoay ±5°/±10°, resize 0.8/1.2, sáng 0.8/1.2,
Gauss σ=10, muối tiêu 2%, JPEG 30/50/70, blur 1.5, crop 5% mỗi phía.
Xoay giữ nguyên khung, phần trống nền trắng; resize thay đổi độ phân giải, không mô phỏng zoom camera.

Cặp dương: ảnh gốc và biến thể cùng nguồn. Cặp âm: mỗi tổ hợp hai nguồn khác nhau một lần.
Nhãn dựa trên nguồn gốc ảnh, không được suy ra từ khoảng cách hash.
""")
code("""if not (DATA / 'dataset_pairs.json').exists():
    build_dataset(DATA, seed=42)
manifest = json.loads((DATA / 'dataset_pairs.json').read_text(encoding='utf-8'))
positives = sum(p['label'] for p in manifest['pairs'])
print(f"Ảnh gốc: {len(manifest['originals'])}; cặp tương tự: {positives}; "
      f"cặp khác biệt: {len(manifest['pairs']) - positives}")
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
for ax, item in zip(axes.flat, manifest['originals']):
    with Image.open(DATA / item['path']) as im:
        ax.imshow(im)
    ax.set_title(item['id'] + (' (tự vẽ)' if item['synthetic'] else ''))
    ax.axis('off')
fig.suptitle('15 ảnh nguồn — Trần Ngọc Nhân')
fig.tight_layout()
fig.savefig(OUT / 'originals.png', bbox_inches='tight')
plt.show()
""")
code("""examples = [p for p in manifest['pairs'] if p['source_id1'] == 'astronaut' and p['label'] == 1]
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, pair in zip(axes.flat, examples):
    with Image.open(DATA / pair['image2']) as im:
        ax.imshow(im)
    ax.set_title(pair['transform'])
    ax.axis('off')
fig.suptitle('15 biến dạng từ cùng ảnh nguồn')
fig.tight_layout()
fig.savefig(OUT / 'augmentations.png', bbox_inches='tight')
plt.show()
print(json.dumps(examples[0], ensure_ascii=False, indent=2))
""")
md("""## 2. Tiền xử lý và DWT — các mục 2.1–2.2

Ảnh được sửa hướng EXIF, ghép alpha lên nền trắng, chuyển xám, resize 32×32,
đưa về [0,1]. Haar level 2 với `periodization` cho LL kích thước 8×8.
`wavedec2` trả `[cA_n, (cH_n,cV_n,cD_n), ..., (cH_1,cV_1,cD_1)]`.
Ở đây hiển thị **bốn băng cùng cấp sâu nhất**: LL=cA, LH=cH, HL=cV, HH=cD.
Tên hướng ngang/dọc tuân theo trục của PyWavelets.

Nguồn: [PyWavelets — DWT 2D](https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html).
""")
code("""hasher = WaveletHasher(hash_size=8, image_size=32, wavelet='haar', level=2)
sample = DATA / 'original/astronaut.png'
analysis = hasher.analyze(sample)
fig, axes = plt.subplots(1, 5, figsize=(13, 3))
for ax, key in zip(axes, ['gray', 'LL', 'LH', 'HL', 'HH']):
    matrix = analysis[key]
    limit = max(float(np.abs(matrix).max()), 1e-12)
    if key in ('LH', 'HL', 'HH'):
        ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit)
    else:
        ax.imshow(matrix, cmap='gray')
    ax.set_title(f'{key}: {matrix.shape}')
    ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'subbands.png', bbox_inches='tight')
plt.show()
restored = pywt.waverec2(analysis['coeffs'], 'haar', mode='periodization')
print('Sai số tái dựng lớn nhất:', np.max(np.abs(restored - analysis['gray'])))
""")
md("""## 3. Lượng tử hóa và sinh wHash — các mục 2.3 và 2.5

Bit = 1 nếu hệ số **≥ ngưỡng median/mean**, ngược lại bằng 0. Đọc theo từng hàng.
Với LL lớn hơn kích thước hash, lấy trung bình các khối đều nhau; LL nhỏ hơn bị từ chối.
Để có 256 bit dùng ảnh 64×64, Haar level 2 → LL 16×16.
Cấu hình được lưu cùng hash. Không so sánh hash tạo bằng các cấu hình khác nhau.
Đây là biến thể wHash theo đề bài, không cam kết trùng thư viện ImageHash.
""")
code("""fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].imshow(analysis['features'], cmap='gray')
axes[0].set_title('LL dùng lượng tử hóa')
axes[1].imshow(analysis['bits'].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
axes[1].set_title('64 bit: trắng = 1')
for ax in axes:
    ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'hash_bits.png', bbox_inches='tight')
plt.show()
print('Ngưỡng:', analysis['cutoff'])
print('Nhị phân:', ''.join(analysis['bits'].astype(np.uint8).astype(str)))
print('Hex 64 bit:', analysis['hex'])
for method in ('median', 'mean'):
    large = WaveletHasher(16, 64, threshold=method)
    bits = large.hash(sample)
    print(f'{method}: {len(bits)} bit, hex = {hash_to_hex(bits)}')
""")
md("""## 4. Khoảng cách Hamming — mục 2.4

$D_H$ là số bit lệch; $S=100(1-D_H/N)$ là phần trăm bit trùng nhau,
**không phải xác suất hai ảnh cùng đối tượng**. Chưa đưa ra nhãn dự đoán vì ngưỡng
phải được chọn và kiểm chứng ở Phase 3.
""")
code("""print(f"{'Biến dạng':<22} {'Hamming':>8} {'Bit trùng (%)':>15}")
for pair in examples:
    comparison = hasher.compare(DATA / pair['image1'], DATA / pair['image2'])
    print(f"{pair['transform']:<22} {comparison['distance']:>8} {comparison['similarity']:>15.2f}")
print('So với ảnh coffee:', hasher.compare(sample, DATA / 'original/coffee.png'))
assert hamming_distance('0101', '0011') == 2
assert similarity_percentage('0101', '0011') == 50
""")
md("""## 5. Bàn giao cho Phase 3 và giới hạn

Xuất hash và khoảng cách cho toàn bộ cặp ảnh để Thanh Nguyên dùng đánh giá.
Các cặp không độc lập vì dùng chung ảnh nguồn. Khi chia tập chọn ngưỡng/kiểm thử,
**chia theo `source_id` trước**, rồi tạo/lọc cặp chỉ chứa nguồn trong cùng tập;
không chia ngẫu nhiên từng cặp gây rò rỉ dữ liệu. Cặp âm có hai nguồn nên phải kiểm tra cả hai.

Ảnh đơn sắc có thể trùng hash; ảnh khác nhau cũng có thể va chạm hash.
wHash không bất biến hoàn toàn với xoay, crop hay góc chụp. Các chữ số cùng nền trắng
có thể gần nhau về hash dù khác nguồn. Bộ dữ liệu 15 nguồn chỉ phù hợp kiểm tra pipeline.
""")
code("""result = compute_pair_distances(DATA, hasher)
(OUT / 'pair_distances.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
for label, name in [(1, 'Cùng nguồn'), (0, 'Khác nguồn')]:
    values = np.array([p['distance'] for p in result['pairs'] if p['label'] == label])
    print(f'{name}: n={values.size}, min={values.min()}, trung bình={values.mean():.2f}, max={values.max()}')
print('Đã lưu:', OUT / 'pair_distances.json')
""")
md("""## 6. Kiểm thử

Chạy `python -m unittest -v test_vision_wavelet.py` tại thư mục lab3.
Đối chứng gồm Haar với ma trận biết trước, tái dựng nghịch đảo trên 6 họ wavelet,
64/256 bit, median/mean, Hamming, alpha, ảnh đơn sắc, tham số sai,
tái lập nhiễu và kiểm tra mọi cặp/ảnh trong chỉ mục.

Nguồn ảnh và mô tả: [scikit-image data](https://scikit-image.org/docs/stable/api/skimage.data.html),
chi tiết từng ảnh được lưu trong `dataset_pairs.json`.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
nbf.write(notebook, root / 'Lab3.ipynb')
print(root / 'Lab3.ipynb')
