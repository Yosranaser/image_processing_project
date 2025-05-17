import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️filters on images app")

# رفع الصورة
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])

# اختيار الفلتر
filter_option = st.selectbox("اختر الفلتر:", ["-- اختر --", "Grayscale", "Blur", "Edge Detection", "Sepia","salt and pepper noise","gaussian_noise"])

# تطبيق الفلتر
if uploaded_file is not None and filter_option != "-- اختر --":
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # تحويل الصورة إلى BGR لأن OpenCV يعمل بـ BGR
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if filter_option == "Grayscale":
        filtered_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        st.image(filtered_img, caption="صورة رمادية", use_column_width=True)
    elif filter_option == "gaussian_noise":
        def add_gaussian_noise(image,mean,std):
            noise = np.random.normal(mean,std,image.shape).astype(np.uint8)
            noisy_image = cv2.add(noise,image)
            st.image(noisy_image, caption="صورة رمادية", use_column_width=True)
        add_gaussian_noise(image,0,25)
       
    elif filter_option == "Blur":
        filtered_img = cv2.GaussianBlur(img_bgr, (15, 15), 0)
        st.image(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB), caption="صورة ضبابية", use_column_width=True)

    elif filter_option == "Edge Detection":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.Canny(gray, 100, 200)
        st.image(filtered_img, caption="اكتشاف الحواف", use_column_width=True)

    
