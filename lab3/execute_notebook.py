"""Chạy notebook từ đầu, lưu đầu ra và kiểm tra không có ô lỗi."""
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
# Giữ các tệp tạm của Jupyter và Matplotlib trong thư mục bài thực hành.
runtime = root / '.runtime'
runtime.mkdir(exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime)
os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
notebook = nbformat.read(root / 'Lab3.ipynb', as_version=4)
NotebookClient(notebook, timeout=180, kernel_name='python3',
               resources={'metadata': {'path': str(root)}}).execute()
nbformat.validate(notebook)
assert not any(output.output_type == 'error' for cell in notebook.cells
               if cell.cell_type == 'code' for output in cell.outputs)
nbformat.write(notebook, root / 'Lab3.ipynb')
print('Notebook OK:', sum(cell.cell_type == 'code' for cell in notebook.cells), 'code cells')
