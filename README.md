# ✨ Image Segmentation & Object Detection using K-Means and Streamlit

This project is a Streamlit-based web application that allows users to upload an image and apply K-Means clustering for segmentation. It detects and highlights the most prominent object based on color thresholding in the HSV color space. The app is fully interactive with controls for adjusting clusters and HSV ranges.

## 🚀 Features

- Upload any JPG/PNG image from your local system
- Interactive sidebar to control:
  - K value (number of clusters)
  - Which cluster to highlight
  - HSV color thresholds
- View image processing stages in organized tabs:
  - Segmented Image
  - Mask Highlight
  - HSV Threshold
  - Detected Objects
- Automatic bounding box on the largest detected object
- Stylish UI with custom CSS for better user experience
- Download processed output image

## 🧠 How It Works

1. Image is uploaded and resized for performance.
2. K-Means clustering is applied to segment the image.
3. The user-selected cluster is highlighted using color masking.
4. Masked image is converted to HSV for better color filtering.
5. HSV thresholds are applied to isolate the object.
6. The largest contour is detected and bounded with a box.
7. Results are shown with clear visuals in four tabs.

## 🛠 Tech Stack

- Python 3.x
- Streamlit
- OpenCV (cv2)
- NumPy
- Pillow (PIL)

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/kmeans-object-detection.git
cd kmeans-object-detection
