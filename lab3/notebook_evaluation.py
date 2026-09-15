"""Các ô notebook Phase 3–4; dùng chung khi tạo mới và bổ sung bản đã có."""
import nbformat as nbf


def evaluation_cells():
    """Trả phần báo cáo Thanh Nguyên, tính kết quả từ dataset của Nhân."""
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    return [
        md('''## 7. Đánh giá hiệu suất — Thanh Nguyên (Phase 3)

Tái sử dụng dataset, `WaveletHasher` và Hamming của Trần Ngọc Nhân.
Quy tắc dự đoán: tương tự nếu **Hamming ≤ θ**; quét θ từ 0 đến N bit.
TP/TN/FP/FN dùng nhãn 1 là cùng nguồn. Recall = TP/(TP+FN),
Specificity = TN/(TN+FP), Precision = TP/(TP+FP),
Accuracy = (TP+TN)/n, F1 = 2TP/(2TP+FP+FN). Mẫu số 0 trả 0.

Chia ngẫu nhiên **theo nguồn** với seed 42: 10 nguồn hiệu chỉnh, 5 nguồn kiểm thử.
Cặp âm nối hai tập được loại. Chọn θ trên tập hiệu chỉnh bằng
**Youden J = Recall + Specificity − 1**; hòa chọn θ nhỏ nhất.
Áp dụng nguyên θ này lên kiểm thử. ROC dùng điểm **−Hamming**, không dùng nhãn dự đoán.
Kết quả toàn bộ 330 cặp chỉ là thống kê mô tả, không phải kiểm thử độc lập.

Nguồn: [ROC scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html),
[giới hạn cấp DWT PyWavelets](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html).
'''),
        code('''from vision_wavelet import run_evaluation
report = run_evaluation(DATA, OUT / 'evaluation')
print(json.dumps(report['split'], ensure_ascii=False, indent=2))
default = report['methods']['whash_default']
print('AUC kiểm thử:', default['test_auc'])
print(json.dumps(default['test_metrics'], indent=2))
from IPython.display import display, Image as DisplayImage, Markdown
display(DisplayImage(filename=str(OUT / 'evaluation/evaluation.png')))
matrix = np.load(OUT / 'evaluation/hamming_matrix.npy')
assert matrix.shape == (240, 240)
assert np.array_equal(matrix, matrix.T) and not matrix.diagonal().any()
print('Ma trận Hamming toàn bộ ảnh:', matrix.shape)
'''),
        md('''## 8. Khảo sát phương pháp băm — Thanh Nguyên (Phase 4)

Khảo sát 6 họ wavelet × 4 level × 2 kích thước hash = 48 cấu hình,
thêm wHash mặc định của Nhân và aHash/dHash/pHash 64/256 bit: **55 cấu hình**.
48 cấu hình wHash dùng chung ảnh **256×256**: đủ level 4 cho `coif2` và
LL ≥ 16×16, giúp so sánh level mà không thay kích thước ảnh đầu vào.
LL lớn được lấy trung bình theo khối bằng chính API của Nhân.

aHash so mức xám với mean; dHash so pixel phải ≥ pixel trái;
pHash dùng DCT-II trực chuẩn trên ảnh 4h×4h, lấy vùng h×h tần số thấp,
bỏ DC khi tính median nhưng giữ bit DC. Các bit aHash/pHash dùng ≥ ngưỡng.
Đây là các biến thể được định nghĩa trong mã, không cam kết trùng ImageHash.

Mỗi cấu hình chọn ngưỡng trên cùng tập hiệu chỉnh và báo cáo AUC/metrics trên
cùng tập kiểm thử. Không dùng thứ hạng kiểm thử để tuyên bố một cấu hình tối ưu tổng quát.
Thời gian: warm-up một lượt, median 3 lượt trên 240 ảnh PIL đã giải mã;
gồm tiền xử lý và băm, không gồm đọc tệp. Kích thước tiền xử lý khác nhau giữa
các phương pháp nên thời gian phản ánh toàn pipeline, không chỉ phép biến đổi.
'''),
        code('''import csv
with (OUT / 'evaluation/benchmark.csv').open(encoding='utf-8-sig') as stream:
    benchmark_rows = list(csv.DictReader(stream))
columns = ['name', 'n_bits', 'threshold', 'accuracy', 'recall', 'specificity', 'precision', 'f1', 'auc', 'latency_ms']
text = '| ' + ' | '.join(columns) + ' |\\n| ' + ' | '.join(['---'] * len(columns)) + ' |\\n'
text += '\\n'.join('| ' + ' | '.join(str(round(float(r[k]), 4)) if k != 'name' else r[k] for k in columns) + ' |' for r in benchmark_rows)
display(Markdown(text))
display(DisplayImage(filename=str(OUT / 'evaluation/benchmark.png')))
'''),
        md('''### 8.1. Độ bền theo biến dạng và diễn giải kết quả

Chỉ dùng cặp dương trong tập kiểm thử để tính recall theo từng biến dạng tại
ngưỡng đã chọn. Không gọi đây là accuracy vì nhóm này không có cặp âm.
Khoảng cách chia số bit cho phép so sánh 64/256 bit. CSV lưu riêng 15 loại
biến dạng, gồm noise, rotation, scale, lighting, JPEG, blur và crop.
'''),
        code('''with (OUT / 'evaluation/robustness.csv').open(encoding='utf-8-sig') as stream:
    robustness_rows = list(csv.DictReader(stream))
baseline_robustness = [r for r in robustness_rows if r['name'] == 'whash_default']
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar([r['transform'] for r in baseline_robustness], [float(r['recall']) for r in baseline_robustness])
ax.set(ylabel='Recall trên kiểm thử', ylim=(0, 1), title='wHash mặc định: độ bền theo biến dạng')
ax.tick_params(axis='x', labelrotation=65)
fig.tight_layout()
fig.savefig(OUT / 'evaluation/robustness.png', dpi=140)
plt.show()
m = default['test_metrics']
weakest = min(baseline_robustness, key=lambda r: float(r['recall']))
display(Markdown(f"wHash mặc định chọn **θ={m['threshold']}** trên hiệu chỉnh; "
    f"kiểm thử có **AUC={default['test_auc']:.4f}**, Accuracy={m['accuracy']:.4f}, "
    f"Recall={m['recall']:.4f}, Specificity={m['specificity']:.4f}, F1={m['f1']:.4f}. "
    f"Một biến dạng có recall thấp nhất là `{weakest['transform']}` ({float(weakest['recall']):.2f})."))
print('Môi trường:', report['environment'])
'''),
        md('''### 8.2. Giới hạn và kiểm chứng

Chỉ có 15 nguồn, gồm 5 chữ số tổng hợp; các biến thể cùng nguồn không độc lập.
Một phép chia seed 42 có phương sai lớn; kết quả chỉ minh họa pipeline,
không đại diện cho nhận diện đối tượng hoặc ảnh ngoài bộ dữ liệu.
Thay đổi level Haar kèm block averaging có thể tạo đặc trưng tương đương;
không diễn giải các hàng trùng kết quả là lỗi hay chứng minh level không có tác dụng.

Chạy `python -m unittest -v test_vision_wavelet test_evaluation` để kiểm chứng:
ma trận nhầm lẫn biết trước, AUC hoàn hảo/đảo ngược/tied scores, đầu mút ROC,
ngưỡng hòa, chia nguồn không rò rỉ, gradient dHash và tính trực chuẩn DCT.
Phase 5 và phần tích hợp toàn nhóm Phase 6 vẫn theo phân công gốc.
''')]


if __name__ == '__main__':
    from pathlib import Path
    root = Path(__file__).resolve().parent
    notebook = nbf.read(root / 'Lab3.ipynb', as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'markdown' and cell.source.startswith('## 7. Đánh giá hiệu suất'):
            notebook.cells = notebook.cells[:index]
            break
    notebook.cells.extend(evaluation_cells())
    nbf.validate(notebook)
    nbf.write(notebook, root / 'Lab3.ipynb')
