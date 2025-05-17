import streamlit as st
import cv2
import numpy as np
from PIL import Image



def add_gaussian_noise(image, mean=0, std=25):
    image_np = np.array(image)  # Convert PIL image to NumPy array
    noise = np.random.normal(mean, std, image_np.shape).astype(np.uint8)
    noisy_image = image_np + noise
    noisy_image = np.clip(noisy_image, 0, 255)  
    return Image.fromarray(noisy_image.astype(np.uint8))
    
def add_salt_and_paper_noise(image, noisy_ratio):
    noisy_image = image.copy()
    h,w,c = noisy_image.shape
    noisy_pixels = int( h * w* noisy_ratio)
    for _ in range (noisy_pixels):
        row,colm = np.random.randint(0,h), np.random.randint(0,w)
        if np.random.rand() < 0.5:
            noisy_image[row,colm] = [0,0,0]
        else:
            noisy_image[row,colm] = [255,255,255]
    return noisy_image

def add_random_noise(image,intensity):
    noisy_image = image.copy()
    noise = np.random.randint(-1*intensity, intensity +1,image.shape)
    noisy_image = np.clip((image + noisy_image), 0, 255).astype(np.uint8)
    return noisy_image
def image_compression(image,comp_ratio):
    h,w = image.size
    row,colm =int( h/comp_ratio),int(w/comp_ratio)
    new_size = (row,colm)
    resized_image = image.resize(new_size)
    resized_image.save("comp.jpg",optimize = False,quality = 50)
    comp_size = os.path.getsize("comp.jpg")
    print("comp size : ",comp_size)
    
st.title("🖼️filters on images app")


uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])


filter_option = st.selectbox("اختر الفلتر:", ["-- اختر --","Grayscale", "Blur", "Edge Detection","salt and pepper noise","gaussian_noise","random_noise","image_compression"])


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
    elif filter_option == "salt and pepper noise":
         noisy_img = add_salt_and_paper_noise(img_bgr,0.5)
         st.image(noisy_img, caption="صورة بها ضوضاء salt and pepper", use_column_width=True)
    elif filter_option == "random_noise":   
         filtered_img = add_random_noise(img_bgr,100)
         st.image(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB), caption="صورة بضوضاء عشوائيه ", use_column_width=True)
    elif filter_option == "Blur":
        filtered_img = cv2.GaussianBlur(img_bgr, (15, 15), 0)
        st.image(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB), caption="صورة ضبابية", use_column_width=True)
    elif filter_option == "image_compression":
        image_compression(img_bgr,2)
    elif filter_option == "Edge Detection":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.Canny(gray, 100, 200)
        st.image(filtered_img, caption="اكتشاف الحواف", use_column_width=True)

    
