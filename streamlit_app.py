import streamlit as st
import cv2
import numpy as np
from PIL import Image


def add_gaussian_noise(image, mean=0, std=25):
    image_np = np.array(image)  # Convert PIL image to NumPy array
    noise = np.random.normal(mean, std, image_np.shape).astype(np.uint8)
    noisy_image = image_np + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
    return Image.fromarray(noisy_image.astype(np.uint8))

st.title("🖼️filters on images app")


uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])


filter_option = st.selectbox("اختر الفلتر:", ["-- اختر --","Grayscale", "Blur", "Edge Detection","salt and pepper noise","gaussian_noise"])


if uploaded_file is not None and filter_option != "-- اختر --":
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if filter_option == "Grayscale":
        filtered_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        st.image(filtered_img, caption="صورة رمادية", use_column_width=True)
    elif filter_option == "gaussian_noise":
        image = image.convert("RGB")
        noisy_img = add_gaussian_noise(img_bgr, 0, 25)
        st.image(noisy_img, caption="صورة بها ضوضاء Gaussian", use_column_width=True)

       
    elif filter_option == "Blur":
        filtered_img = cv2.GaussianBlur(img_bgr, (15, 15), 0)
        st.image(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB), caption="صورة ضبابية", use_column_width=True)

    elif filter_option == "Edge Detection":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.Canny(gray, 100, 200)
        st.image(filtered_img, caption="اكتشاف الحواف", use_column_width=True)

    
