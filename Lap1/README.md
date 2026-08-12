# 1. Introduction

This project covers basic image processing with Python, OpenCV, and Pillow.

The main tasks are:

- Read, display, and save an image.
- Convert an image to grayscale, HSV, and LAB.
- Crop and resize an image.
- Draw lines, circles, and rectangles.
- Add text to an image.

# 2. How to Setup

1. Open a terminal and enter the `Lap1` directory:

   ```bash
   cd Lap1
   ```

2. Install the required libraries:

   ```bash
   pip install opencv-python Pillow matplotlib jupyter
   ```

3. Create the image folders if they do not exist:

   ```bash
   mkdir images/input images/output
   ```

4. Place an image at:

   ```text
   images/input/sample.jpg
   ```

5. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open `Lap1.ipynb` and run the cells in order.

# 3. How to Implement

Write all code in `Lap1.ipynb` and complete these phases:

1. Install and import OpenCV and Pillow.
2. Read, display, and save an image in a different format.
3. Convert the image to grayscale, HSV, and LAB.
4. Crop the image and resize it by scale or fixed dimensions.
5. Draw a line, circle, rectangle, and text on the image.

## Project Directory

```text
Lap1/
├── README.md
├── Lap1.ipynb
├── Doc/
│   ├── Requirement.md
│   ├── plan.md
│   └── status.md
└── images/
    ├── input/
    │   └── sample.jpg
    └── output/
        ├── converted_image.png
        ├── grayscale.jpg
        ├── hsv.jpg
        ├── lab.jpg
        ├── cropped.jpg
        ├── resized.jpg
        └── drawing.jpg
```
