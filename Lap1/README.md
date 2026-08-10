# 1. Introduction

This project introduces basic image-processing techniques using **OpenCV** and **Pillow** in Python.

The project focuses on practical image operations while also explaining the technical concepts behind each operation.

The main tasks include:

- Reading, displaying, and saving images.
- Converting image color spaces.
- Cropping and resizing images.
- Drawing shapes and adding text.

---

# 2. Technologies

## 2.1 Python

Python is used as the main programming language because it provides strong support for Computer Vision and image-processing libraries.

## 2.2 OpenCV

OpenCV is the main library used for:

- Reading images.
- Displaying images.
- Color conversion.
- Resizing.
- Cropping.
- Drawing.
- Computer Vision processing.

Installation:

```bash
pip install opencv-python
```

## 2.3 Pillow

Pillow is used as an additional image-processing library.

It supports:

- Opening images.
- Saving images.
- Image-format conversion.
- Resizing.
- Cropping.
- Basic image manipulation.

Installation:

```bash
pip install Pillow
```

---

# 3. Task 1 – Read, Display, and Save Images

## 3.1 What do we do?

In this task, we will:

- Read an image from the computer.
- Verify that the image is loaded correctly.
- Display the image on the screen.
- Save the image using another file format.

Example:

```text
sample.jpg
    ↓
Read Image
    ↓
Display Image
    ↓
Save Image
    ↓
sample.png
```

---

## 3.2 Technical Background

A digital image can be represented as a matrix of pixels.

Each pixel contains information about its color or brightness.

For a normal color image, each pixel usually contains three channels:

```text
Red
Green
Blue
```

When OpenCV reads an image, the image is stored as a multidimensional array.

For example:

```text
Height × Width × Channels
```

A 1920 × 1080 color image can therefore be represented as:

```text
1080 × 1920 × 3
```

OpenCV reads color images using the **BGR channel order** by default.

---

## 3.3 Why do we need this task?

Reading an image is the first step in almost every Computer Vision application.

Before an image can be:

- Analyzed.
- Converted.
- Cropped.
- Resized.
- Detected by an AI model.

the program must first load it into memory.

Saving images is also important because processed results often need to be stored for later use.

---

## 3.4 Where / When is it applied?

This operation is used in:

- Image-processing applications.
- Object Detection.
- Face Recognition.
- Medical Imaging.
- Traffic Camera systems.
- Dataset preparation.

Example:

```text
Camera
   ↓
Capture Image
   ↓
Read Image
   ↓
Process Image
   ↓
Save Result
```

---

## 3.5 How do we implement it?

```python
import cv2

image = cv2.imread("images/input/sample.jpg")

if image is None:
    print("Image cannot be loaded.")
else:
    cv2.imshow("Original Image", image)

    cv2.imwrite(
        "images/output/converted_image.png",
        image
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

Main functions:

```python
cv2.imread()
cv2.imshow()
cv2.imwrite()
```

---

## 3.6 Expected Result

Input:

```text
sample.jpg
```

Output:

```text
converted_image.png
```

The original image should also be displayed successfully.

---

# 4. Task 2 – Color Space Conversion

## 4.1 What do we do?

In this task, we convert the original image into several different color spaces:

- Grayscale.
- HSV.
- LAB.

Processing flow:

```text
             Original Image
                    ↓
               Read Image
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
   Grayscale       HSV         LAB
        ↓           ↓           ↓
       Save        Save        Save
```

---

# 4.2 Technical Background

## 4.2.1 RGB and BGR

Most digital images are commonly represented using:

```text
RGB
```

where:

- R = Red.
- G = Green.
- B = Blue.

However, OpenCV uses:

```text
BGR
```

by default.

Therefore, an image read using:

```python
cv2.imread()
```

normally follows this channel order:

```text
Blue → Green → Red
```

Understanding this is important when converting images between OpenCV and other libraries.

---

## 4.2.2 Grayscale

A grayscale image contains only brightness information.

Instead of three color channels, it normally uses one intensity channel.

Typical values range from:

```text
0   → Black
255 → White
```

### Why use Grayscale?

Grayscale reduces the amount of information that needs to be processed.

Instead of:

```text
Height × Width × 3
```

we process:

```text
Height × Width
```

This can reduce computational complexity.

### Where is Grayscale applied?

- Edge Detection.
- OCR.
- Face Detection.
- Shape Detection.
- License Plate Recognition.

### Implementation

```python
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)
```

---

## 4.2.3 HSV

HSV represents colors using:

- H = Hue.
- S = Saturation.
- V = Value.

### Hue

Represents the actual color.

Examples:

```text
Red
Green
Blue
Yellow
```

### Saturation

Represents the intensity or purity of the color.

### Value

Represents the brightness.

### Why use HSV?

HSV separates color information from brightness.

This makes it useful when detecting objects based mainly on their colors.

### Where is HSV applied?

- Color Detection.
- Object Tracking.
- Traffic Sign Detection.
- Vehicle Color Detection.
- Image Segmentation.

### Implementation

```python
hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)
```

---

## 4.2.4 LAB

LAB consists of:

- L = Lightness.
- A = Green ↔ Red.
- B = Blue ↔ Yellow.

### Why use LAB?

LAB separates brightness from color information.

It can therefore be useful when illumination conditions change.

### Where is LAB applied?

- Image Enhancement.
- Color Correction.
- Image Segmentation.
- Medical Imaging.
- Lighting Normalization.

### Implementation

```python
lab = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2LAB
)
```

---

## 4.3 Why do we perform color conversion?

Different Computer Vision problems require different types of image information.

For example:

```text
Need brightness/shape
        ↓
    Grayscale
```

```text
Need color detection
        ↓
       HSV
```

```text
Need color + lighting analysis
        ↓
       LAB
```

Therefore, choosing an appropriate color space can simplify later image-processing tasks.

---

## 4.4 Expected Result

The program should generate:

```text
grayscale.jpg
hsv.jpg
lab.jpg
```

---

# 5. Task 3 – Cropping Images

## 5.1 What do we do?

In this task, we select and extract a specific area from an image.

Example:

```text
Original Image

+------------------------+
|                        |
|      +----------+      |
|      | Vehicle  |      |
|      +----------+      |
|                        |
+------------------------+

             ↓

Cropped Image

+----------+
| Vehicle  |
+----------+
```

---

## 5.2 Technical Background

An image in OpenCV is stored as an array.

Therefore, a specific section can be accessed using array slicing.

The basic structure is:

```python
image[y1:y2, x1:x2]

To crop an image, we normally define two corner points.

(x1, y1) ────────────────────
    |                       |
    |                       |
    |         ROI           |
    |                       |
    |                       |
    └────────────────── (x2, y2)

Where:

x1 = left boundary.
y1 = top boundary.
x2 = right boundary.
y2 = bottom boundary.

The width of the cropped region is:

ROI Width = x2 - x1

The height is:

ROI Height = y2 - y1

For example:

(x1, y1) = (200, 100)
(x2, y2) = (600, 400)

Then:

Width  = 600 - 200 = 400 pixels
Height = 400 - 100 = 300 pixels

Therefore, the cropped image will have approximately:

400 × 300

pixels.
```



---

## 5.3 Why do we crop images?

Cropping allows us to remove unnecessary image information and focus only on the important region.

This can:

- Reduce processing.
- Reduce noise.
- Focus analysis on an object.
- Prepare detected objects for another model.

---

## 5.4 Where / When is cropping applied?

Cropping is often used in:

- Face Recognition.
- License Plate Recognition.
- Vehicle Detection.
- OCR.
- Object Tracking.

Example:

```text
Camera Image
     ↓
Detect Car
     ↓
Find Bounding Box
     ↓
Crop Car Region
     ↓
Analyze Car
```

---

## 5.5 How do we implement it?

```python
cropped = image[100:400, 200:600]

cv2.imshow("Cropped Image", cropped)

cv2.imwrite(
    "images/output/cropped.jpg",
    cropped
)
```

---

## 5.6 Expected Result

```text
cropped.jpg
```

should contain only the selected region from the original image.

---

# 6. Task 4 – Resize Images

## 6.1 What do we do?

We change the width and height of an image.

Example:

```text
1920 × 1080
      ↓
Resize
      ↓
640 × 480
```

Two methods will be tested:

- Fixed-size resizing.
- Scale-based resizing.

---

## 6.2 Technical Background

Image resolution is usually represented as:

```text
Width × Height
```

For example:

```text
1920 × 1080
```

contains more pixels than:

```text
640 × 480
```

A larger image generally requires:

- More memory.
- More storage.
- More processing time.

When resizing images, OpenCV calculates new pixel values through an interpolation method.

---

## 6.3 Why do we resize images?

Resizing is useful because many Computer Vision and AI systems require a specific input resolution.

It can also:

- Reduce computational cost.
- Improve processing speed.
- Reduce storage requirements.
- Standardize dataset dimensions.

---

## 6.4 Where / When is resizing applied?

- Object Detection.
- Deep Learning.
- Image Classification.
- Face Recognition.
- Web image optimization.
- Dataset preprocessing.

Example:

```text
Camera Image
1920 × 1080
      ↓
Resize
      ↓
640 × 640
      ↓
AI Model
```

---

## 6.5 How do we implement it?

### Fixed Size

```python
resized = cv2.resize(
    image,
    (640, 480)
)
```

### Scale-Based Resize

```python
scale = 0.5

width = int(image.shape[1] * scale)
height = int(image.shape[0] * scale)

resized = cv2.resize(
    image,
    (width, height)
)
```

---

## 6.6 Expected Result

```text
resized.jpg
```

should have a different resolution while preserving the image content.

---

# 7. Task 5 – Draw Shapes and Add Text

## 7.1 What do we do?

In this task, we add graphical information to the image.

We will draw:

- A line.
- A rectangle.
- A circle.
- Text.

Example:

```text
Original Image
      ↓
Draw Rectangle
      ↓
Draw Circle
      ↓
Add Label
      ↓
Output Image
```

---

## 7.2 Technical Background

Drawing functions are important in Computer Vision because processing results often need to be visualized.

For example, an Object Detection model may return coordinates such as:

```text
x1, y1, x2, y2
```

These coordinates can be used to draw a rectangle around the detected object.

This rectangle is commonly called a:

```text
Bounding Box
```

Text can then be added to display information such as:

```text
Car
Confidence: 95%
```

---

## 7.3 Why do we draw on images?

Drawing allows us to visualize:

- Detected objects.
- Object locations.
- Tracking points.
- Regions of Interest.
- Labels.
- Prediction confidence.

Without visualization, model outputs may only contain numbers or coordinates that are difficult for users to understand.

---

## 7.4 Where / When is it applied?

Common applications include:

- Object Detection.
- Face Detection.
- Vehicle Tracking.
- Pose Estimation.
- Traffic Monitoring.
- Surveillance Systems.

Example:

```text
AI Model
   ↓
Detect Vehicle
   ↓
Bounding Box Coordinates
   ↓
Draw Rectangle
   ↓
Add "Car 95%"
   ↓
Display Result
```

---

## 7.5 How do we implement it?

### Line

```python
cv2.line(
    image,
    (50, 50),
    (400, 50),
    (255, 0, 0),
    3
)
```

### Rectangle

```python
cv2.rectangle(
    image,
    (100, 100),
    (400, 300),
    (0, 255, 0),
    3
)
```

### Circle

```python
cv2.circle(
    image,
    (500, 300),
    80,
    (0, 0, 255),
    3
)
```

### Text

```python
cv2.putText(
    image,
    "Computer Vision",
    (100, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)
```

---

## 7.6 Expected Result

The output image should contain:

```text
Line
Rectangle
Circle
Text
```

and be saved as:

```text
drawing.jpg
```

---

# 8. Pillow Usage

Pillow will be used as an additional library to demonstrate basic image manipulation.

Example:

```python
from PIL import Image

image = Image.open(
    "images/input/sample.jpg"
)

image.show()

image.save(
    "images/output/pillow_image.png"
)
```

Pillow can also perform:

- Resizing.
- Cropping.
- Grayscale conversion.
- Format conversion.

For this project, **OpenCV remains the primary library**, while Pillow provides an alternative approach for basic image operations.

---

# 9. Project Directory Structure

```text
opencv-basic-image-processing/
│
├── README.md
├── requirements.txt
│
├── images/
│   ├── input/
│   │   └── sample.jpg
│   │
│   └── output/
│       ├── converted_image.png
│       ├── grayscale.jpg
│       ├── hsv.jpg
│       ├── lab.jpg
│       ├── cropped.jpg
│       ├── resized.jpg
│       └── drawing.jpg
│
├── src/
│   ├── 01_read_display_save.py
│   ├── 02_color_conversion.py
│   ├── 03_crop_resize.py
│   └── 04_draw_shapes_text.py
│
└── screenshots/
    └── ...
```

---

# 10. Overall Processing Flow

```text
                    Input Image
                         ↓
                    Read Image
                         ↓
                 Display Original
                         ↓
              Color Space Conversion
                /        |        \
               ↓         ↓         ↓
         Grayscale      HSV       LAB
                \        |        /
                         ↓
                    Crop Image
                         ↓
                    Resize Image
                         ↓
                Draw Shapes / Text
                         ↓
                     Save Output
                         ↓
                    Output Images
```

---

# 11. Expected Final Outputs

```text
images/output/
│
├── converted_image.png
├── grayscale.jpg
├── hsv.jpg
├── lab.jpg
├── cropped.jpg
├── resized.jpg
└── drawing.jpg
```


