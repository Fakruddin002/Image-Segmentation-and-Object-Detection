import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Image Segmentation and Object Detection")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to a NumPy array
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Resize for performance (optional)
    resized_image = cv2.resize(image_np, (256, 256))
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

    # Reshape image and apply K-means clustering
    pixel_vals = image_rgb.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image_rgb.shape)
    labels_reshape = labels.reshape(image_rgb.shape[0], image_rgb.shape[1])

    # Cluster selection
    cluster = st.slider("Select cluster to highlight", 0, k-1, 3)
    BLUE = [0, 0, 255]
    masked_image = np.copy(image_rgb)
    masked_image[labels_reshape == cluster] = BLUE

    # Convert to HSV and create threshold for blue
    hsv_img = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)

    lower_blue = (100, 100, 100)
    upper_blue = (140, 255, 255)
    COLOR_MIN = np.array(lower_blue, np.uint8)
    COLOR_MAX = np.array(upper_blue, np.uint8)

    # Threshold the image
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)

    # Find contours
    contours, hierarchy = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # Draw rectangles on original image
    result_img = np.copy(image_rgb)
    pad_w, pad_h, pad_x, pad_y = 6, 8, 6, 8
    for cnt in contours:
        cnt = contours[0]  # take only the largest
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result_img, (x - pad_x, y - pad_y), (x + w + pad_w, y + h + pad_h), (255, 0, 0), 2)

    # Display all images in 3 columns x 2 rows
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(image_rgb, channels="RGB")
    with col2:
        st.subheader("Segmented")
        st.image(segmented_image, channels="RGB")
    with col3:
        st.subheader("Masked")
        st.image(masked_image, channels="RGB")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.subheader("HSV Image")
        st.image(hsv_img, channels="HSV")
    with col5:
        st.subheader("Thresholded")
        st.image(frame_threshed, clamp=True)
    with col6:
        st.subheader("Detected Objects")
        st.image(result_img, channels="RGB")

    # Option to download result image
    result_pil = Image.fromarray(result_img)
    buf = BytesIO()
    result_pil.save(buf, format="PNG")
    st.download_button("Download Detected Image", buf.getvalue(), "detected_objects.png", "image/png")