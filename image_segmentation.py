import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Custom CSS for nicer background and styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 24px 0 rgba(0,0,0,0.2);
    }
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
    }
    .stButton>button {
        background-color: #764ba2;
        color: white;
        border-radius: 12px;
        padding: .6em 1.5em;
        font-size: 1rem;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #667eea;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✨ Image Segmentation & Object Detection ✨")

# File uploader with help tooltip
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"], help="Choose an image to segment and detect objects.")

if uploaded_file is not None:
    # Read image and convert to RGB NumPy array
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Resize image for performance
    resized_image = cv2.resize(image_np, (300, 300))
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    st.markdown("### Input Image:")
    st.image(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB), use_column_width=True)

    # User controls in sidebar
    st.sidebar.header("Adjust Parameters")

    k = st.sidebar.slider("Number of K-Means Clusters (K)", min_value=2, max_value=10, value=4, step=1)
    cluster = st.sidebar.slider("Select Cluster to Highlight", min_value=0, max_value=k-1, value=0, step=1)

    # Color pickers for HSV range (default blue-ish range)
    st.sidebar.markdown("### HSV Color Thresholds for Object Detection")

    default_lower = [100, 150, 0]
    default_upper = [140, 255, 255]

    lower_h = st.sidebar.slider("Lower Hue", 0, 179, default_lower[0])
    lower_s = st.sidebar.slider("Lower Saturation", 0, 255, default_lower[1])
    lower_v = st.sidebar.slider("Lower Value", 0, 255, default_lower[2])

    upper_h = st.sidebar.slider("Upper Hue", 0, 179, default_upper[0])
    upper_s = st.sidebar.slider("Upper Saturation", 0, 255, default_upper[1])
    upper_v = st.sidebar.slider("Upper Value", 0, 255, default_upper[2])

    lower_blue = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_blue = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

    # --- K-Means clustering ---
    pixel_vals = image_rgb.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image_rgb.shape)
    labels_reshape = labels.reshape(image_rgb.shape[0], image_rgb.shape[1])

    # Color the selected cluster's pixels bright red in masked image for emphasis
    masked_image = np.copy(segmented_image)
    masked_image[labels_reshape == cluster] = [255, 0, 0]

    # Convert masked image to HSV for thresholding
    hsv_img = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    # Threshold image for selected HSV range
    frame_threshed = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Find contours (objects) from thresholded image
    contours, _ = cv2.findContours(frame_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Copy original image for drawing detections
    result_img = np.copy(image_rgb)

    # Draw bounding boxes and label detected objects
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 100:  # filter out small noise contours by area threshold
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(result_img, f'Object {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Display results in organized tabs
    tabs = st.tabs(["Segmented Image", "Mask Highlight", "HSV Threshold", "Objects Detected"])

    with tabs[0]:
        st.header("Segmented Image")
        st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    with tabs[1]:
        st.header(f"Cluster {cluster} Highlighted")
        st.image(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    with tabs[2]:
        st.header("HSV Threshold Mask")
        st.image(frame_threshed, use_column_width=True)

    with tabs[3]:
        st.header("Detected Objects with Bounding Boxes")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Download processed image button
    st.sidebar.markdown("---")
    if st.sidebar.button("Download Last Detection Result"):
        # Convert to PIL Image
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.sidebar.download_button(label="Download Image", data=byte_im, file_name="detected_objects.png", mime="image/png")

else:
    st.info("Upload an image file to begin segmentation and object detection.")