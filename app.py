import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from io import BytesIO
from skimage.io import imsave

# --- 1️⃣ App title ---
st.title("Image Compression using K-Means Clustering")
st.write("Upload an image, automatically find optimal K, and compress it.")

# --- 2️⃣ File uploader ---
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display original image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original Image", width=600)

    # Convert to NumPy array and flatten
    img_np = np.array(img, dtype=np.float64) / 255.0
    h, w, c = img_np.shape
    pixels = img_np.reshape(-1, 3)

    # --- 3️⃣ Compute inertias for different K values ---
    K_values = [4, 8, 16, 32, 64]
    inertias = []

    st.write("Finding optimal number of clusters (K)...")
    for k in K_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pixels)
        inertias.append(kmeans.inertia_)

    # --- 4️⃣ Find best K using elbow method ---
    knee = KneeLocator(K_values, inertias, curve='convex', direction='decreasing')
    best_k = knee.knee
    st.write(f"Optimal number of clusters detected: **K = {best_k}**")

    # Optional: Show elbow chart
    st.line_chart({"Inertia": inertias}, width=600, height=400)

    # --- 5️⃣ Compress on button click ---
    if st.button("Compress Image"):
        st.write("Compressing image...")
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        labels = kmeans.fit_predict(pixels)
        new_colors = kmeans.cluster_centers_
        compressed_pixels = new_colors[labels]
        compressed_img = (compressed_pixels.reshape(h, w, c) * 255).astype(np.uint8)

        # Display compressed image
        st.image(compressed_img, caption=f"Compressed Image (K={best_k})", width=600)

        # Prepare image for download
        compressed_pil = Image.fromarray(compressed_img)
        buf = BytesIO()
        compressed_pil.save(buf, format="JPEG")
        st.download_button(
            label="Download Compressed Image",
            data=buf,
            file_name=f"compressed_K{best_k}.jpg",
            mime="image/jpeg"
        )
