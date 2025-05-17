import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def add_gaussian_noise(image, mean=0, std=25):
    image_np = np.array(image) 
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

def image_compression(image, factor):
    h, w = image.shape[:2]  
    new_size = (w // factor, h // factor)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    compressed_image = cv2.resize(resized_image, (w, h), interpolation=cv2.INTER_LINEAR)
    return compressed_image

def apply_ideal_high_pass_filter(img, cutoff_freq=2):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    center_row, center_col = rows // 2, cols // 2
    forier_shift = np.fft.fftshift(np.fft.fft2(gray))
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if distance > cutoff_freq:
                mask[i, j] = 1
    ideal_HPF = forier_shift * mask
    filtered_image = np.fft.ifftshift(ideal_HPF)
    filtered_image = np.abs(np.fft.ifft2(filtered_image))
    filtered_image = np.uint8(filtered_image)
    return gray, filtered_image  


def apply_Gaussian_High_pass_filter(img, cutoff_freq):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    crow, ccol = rows // 2 , cols // 2
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    X, Y = np.meshgrid(x, y)
    gaussian_filter = 1 - np.exp(-((X - ccol)**2 + (Y - crow)**2) / (2 * (cutoff_freq**2)))
    filtered_transform = fshift * gaussian_filter
    f_ishift = np.fft.ifftshift(filtered_transform)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.uint8(np.clip(img_back, 0, 255))
    return gray, img_back
def apply_Ideal_Low_pass_filter(img,cutoff_freq):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    center_row, center_col = rows // 2, cols // 2
    cutoff_freq = 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
                distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
                if distance <= cutoff_freq:
                    mask[i, j] = 1
    
    forier=np.fft.fft2(gray)
    forier_shift=np.fft.fftshift(forier)
    ideal_LPF = forier_shift * mask   
    filtered_image = np.fft.ifftshift(ideal_LPF)
    filtered_image = np.abs(np.fft.ifft2(filtered_image))
    filtered_image = np.uint8(filtered_image)
    return gray, filtered_image

st.title("🖼️filters on images app")


uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])


filter_option = st.selectbox("اختر الفلتر:", ["-- اختر --","Grayscale", "Blur", "Edge Detection","salt and pepper noise","gaussian_noise","random_noise","image_compression","ideal_high_pass_filter","Gaussian_High_pass_filter","Ideal_Low_pass_filter"])


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
        filtered_img =image_compression(img_bgr,2)
        st.image(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB), caption="صورة مضغوطة", use_column_width=True)
    elif filter_option == "ideal_high_pass_filter": 
        gray_img, filtered_img = apply_ideal_high_pass_filter(img_bgr,.01)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gray_img, cmap='gray')
        axes[0].set_title("📷 الصورة الأصلية")
        axes[0].axis('off')
        axes[1].imshow(filtered_img, cmap='gray')
        axes[1].set_title("🔍 بعد تطبيق Ideal HPF")
        axes[1].axis('off')
        st.pyplot(fig)
    elif filter_option == "Gaussian_High_pass_filter": 
        gray_img, filtered_img = apply_Gaussian_High_pass_filter(img_bgr,10)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gray_img, cmap='gray')
        axes[0].set_title("📷 orignial image")
        axes[0].axis('off')
        axes[1].imshow(filtered_img, cmap='gray')
        axes[1].set_title("🔍 after applyingt Ideal HPF")
        axes[1].axis('off')
        st.pyplot(fig)
    elif filter_option == "Ideal_Low_pass_filter": 
        gray_img, filtered_img = apply_Ideal_Low_pass_filter(img_bgr,10)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(gray_img, cmap='gray')
        axes[0].set_title("📷 orignial image")
        axes[0].axis('off')
        axes[1].imshow(filtered_img, cmap='gray')
        axes[1].set_title("🔍 after applyingt Ideal HPF")
        axes[1].axis('off')
        st.pyplot(fig)
    elif filter_option == "Edge Detection":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.Canny(gray, 100, 200)
        st.image(filtered_img, caption="اكتشاف الحواف", use_column_width=True)

    
