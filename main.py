import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# Display an image with matplotlib
def display_image(image, caption='Uploaded Image'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(caption)
    st.pyplot(plt)


# Streamlit application
st.title("Object Detection with YOLOv8")
st.write("Upload an image to detect objects")
# File uploader allows the user to upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Button to trigger image analysis
    if st.button("Analyse Image"):
        st.write("Detecting objects...")
        # Perform object detection
        results = model([image])
        # Process results
        result = results[0]
        boxes = result.boxes  # bounding box 
        masks = result.masks  # segmentation masks 
        keypoints = result.keypoints  # Keypoints 
        probs = result.probs  # classification 
        obb = result.obb  # Oriented boxes 
        # Display the results
        result_image = np.squeeze(result.plot())
        display_image(result_image, caption='Detected Objects')
        # Extract the detected objects
        detected_objects = [box.cls_name for box in boxes]
        st.write("Objects detected:")
        for obj in detected_objects:
            st.write(f"- {obj}")



