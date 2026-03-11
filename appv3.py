import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_sauvola, threshold_niblack

# --- Page Config ---
st.set_page_config(page_title="ManuscriptMaster: Web Edition", layout="wide")

st.title("📜 ManuscriptMaster: Research Edition")
st.markdown("---")

# --- Session State for Reset & View Mode ---
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Comparison"

# --- Sidebar Controls ---
st.sidebar.header("🕹️ Controls")

# 1. Load Image
uploaded_file = st.sidebar.file_uploader("📂 Upload Manuscript Image", type=["jpg", "jpeg", "png", "tif", "bmp"])

# 2. Reset Functionality
def reset_params():
    st.session_state.f_strength = 5
    st.session_state.w_size = 25
    st.session_state.e_thresh = 100
    st.session_state.f_type = "Gaussian Blur"
    st.session_state.t_type = "Hybrid (Sauvola+Otsu)"

if st.sidebar.button("🔄 Restore to Originals"):
    reset_params()

# 3. Filtering Section
st.sidebar.subheader("1. Noise Filter Strategy")
filter_type = st.sidebar.selectbox(
    "Filter Type", 
    ("Gaussian Blur", "Non-Local Means", "Median Filter", "Bilateral"),
    key="f_type"
)
filter_strength = st.sidebar.slider("Filter Strength (1-25 px)", 1, 25, key="f_strength")
use_clahe = st.sidebar.checkbox("Enable CLAHE Enhancement", value=True)

# 4. Binarization Section
st.sidebar.subheader("2. Binarization Strategy")
thresh_type = st.sidebar.selectbox(
    "Strategy", 
    ("Hybrid (Sauvola+Otsu)", "Otsu (Global)", "Sauvola (Local)", "Niblack (Local)", "Adaptive Gaussian"),
    key="t_type"
)
window_size = st.sidebar.slider("Local Window Size (3-101 px)", 3, 101, step=2, key="w_size")

# 5. Canny Section
st.sidebar.subheader("3. Canny Sensitivity")
edge_threshold = st.sidebar.slider("Edge Threshold (10-250)", 10, 250, key="e_thresh")

# --- Processing Pipeline ---
def process_manuscript(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    k = filter_strength if filter_strength % 2 != 0 else filter_strength + 1
    if filter_type == "Gaussian Blur":
        processed_gray = cv2.GaussianBlur(gray, (k, k), 0)
    elif filter_type == "Non-Local Means":
        processed_gray = cv2.fastNlMeansDenoising(gray, h=filter_strength)
    elif filter_type == "Median Filter":
        processed_gray = cv2.medianBlur(gray, k)
    elif filter_type == "Bilateral":
        processed_gray = cv2.bilateralFilter(gray, k, 75, 75)
    else:
        processed_gray = gray

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed_gray = clahe.apply(processed_gray)

    w = window_size
    
    if thresh_type == "Otsu (Global)":
        _, binary = cv2.threshold(processed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_type == "Sauvola (Local)":
        binary = (processed_gray > threshold_sauvola(processed_gray, window_size=w)).astype(np.uint8) * 255
    elif thresh_type == "Niblack (Local)":
        binary = (processed_gray > threshold_niblack(processed_gray, window_size=w, k=0.2)).astype(np.uint8) * 255
    elif thresh_type == "Adaptive Gaussian":
        binary = cv2.adaptiveThreshold(processed_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, w, 2)
    else: # Hybrid
        otsu_val, _ = cv2.threshold(processed_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        local_t = threshold_sauvola(processed_gray, window_size=w)
        binary = np.where((processed_gray < local_t) & (processed_gray < otsu_val), 0, 255).astype(np.uint8)

    edges = cv2.Canny(processed_gray, edge_threshold / 2, edge_threshold)
    mask = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    final = np.where((binary == 0) & (mask == 0), 255, binary).astype(np.uint8)
    
    return processed_gray, final

# --- Main Layout ---
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_bgr = cv2.imdecode(file_bytes, 1)
    
    enhanced_gray, final_binary = process_manuscript(original_bgr)
    
    # 📊 Metrics Header
    ref = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    mse = np.mean((ref.astype(np.float32) - final_binary.astype(np.float32)) ** 2)
    psnr = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    score, _ = ssim(ref, final_binary, full=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("MSE", f"{mse:,.1f}")
    m2.metric("PSNR", f"{psnr:,.2f} dB")
    m3.metric("SSIM", f"{score:,.3f}")

    st.markdown("---")

    # 🖼️ Toggle View Mode
    view_choice = st.radio("Display Mode:", ["Comparison View", "Focus: Final Binary"], horizontal=True)

    if view_choice == "Comparison View":
        # Compact side-by-side
        c1, c2, c3 = st.columns(3)
        with c1:
            st.caption("Original")
            st.image(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            st.caption("Enhanced Gray")
            st.image(enhanced_gray, use_container_width=True)
        with c3:
            st.caption("Final Binary")
            st.image(final_binary, use_container_width=True)
    else:
        # Enlarge the processed output
        st.subheader("🔍 Focus: Final Binary Result")
        st.image(final_binary, use_container_width=True)

    # 💾 Download Button
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="💾 Download Result",
        data=cv2.imencode('.png', final_binary)[1].tobytes(),
        file_name="processed_manuscript.png",
        mime="image/png"
    )
else:
    st.info("Please upload an image to start.")
