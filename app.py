import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from datasets import load_dataset
import matplotlib.pyplot as plt
from textblob import TextBlob  # Pure Python alternative for basic grammar correction

# Set page config
st.set_page_config(page_title="Handwriting OCR with Text Improvement", layout="wide")

# Title and description
st.title("✍️ Handwritten Text Recognition")
st.write("Upload an image of handwritten text or explore samples from the IAM dataset")

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    st.error(f"EasyOCR initialization error: {str(e)}")
    st.stop()

# Sidebar for options
with st.sidebar:
    st.header("⚙️ Settings")
    show_preprocessing = st.checkbox("Show preprocessing steps", value=True)
    use_iam_samples = st.checkbox("Use IAM dataset samples", value=False)
    enable_text_correction = st.checkbox("Enable text improvement", value=True)

# Load IAM dataset
@st.cache_resource
def load_iam_data():
    return load_dataset("Teklia/IAM-line")

# Text improvement functions (no Java required)
def improve_text(text):
    if not text.strip():
        return text, []
    
    # Spelling correction
    blob = TextBlob(text)
    corrected = str(blob.correct())
    
    # Basic grammar suggestions (capitalization, punctuation)
    suggestions = []
    if not text[0].isupper():
        suggestions.append("Capitalize first letter")
    if not text.endswith(('.','!','?')):
        suggestions.append("Add ending punctuation")
    
    return corrected, suggestions

# Preprocessing functions
def preprocess_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)
    kernel = np.ones((2,2), np.uint8)
    return cv2.dilate(denoised, kernel, iterations=1)

# Main processing function
def process_image(image):
    try:
        # OCR processing
        results = reader.readtext(np.array(image))
        raw_text = ' '.join([res[1] for res in results]).strip()
        
        # Text improvement
        if enable_text_correction:
            improved_text, suggestions = improve_text(raw_text)
        else:
            improved_text, suggestions = raw_text, []
            
        return {
            'raw_text': raw_text,
            'improved_text': improved_text,
            'suggestions': suggestions,
            'processed_img': preprocess_image(image) if show_preprocessing else None
        }
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None

# Image selection interface
if use_iam_samples:
    iam_dataset = load_iam_data()
    sample_options = [f"Sample {i}" for i in range(10)]
    selected_sample = st.selectbox("Select IAM sample", sample_options)
    sample_idx = int(selected_sample.split()[1])
    sample = iam_dataset['train'][sample_idx]
    st.session_state.current_image = sample['image']
    st.session_state.ground_truth = sample['text']
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.current_image, caption="IAM Dataset Sample", use_column_width=True)
    with col2:
        st.write("**Ground Truth Text:**")
        st.code(st.session_state.ground_truth)
else:
    uploaded_file = st.file_uploader("Upload handwritten text image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.session_state.current_image = Image.open(uploaded_file)
        st.session_state.ground_truth = None
        st.image(st.session_state.current_image, caption="Uploaded Image", use_column_width=True)

# Process and display results
if st.button("🔍 Process Image"):
    if st.session_state.current_image is not None:
        with st.spinner("Processing..."):
            result = process_image(st.session_state.current_image)
            
            if result:
                st.subheader("📝 Results")
                
                tab1, tab2 = st.tabs(["Improved Text", "Details"])
                
                with tab1:
                    st.text_area("Final Text", result['improved_text'], height=150)
                    if enable_text_correction and result['suggestions']:
                        st.write("**Improvements made:**")
                        for suggestion in result['suggestions']:
                            st.write(f"- {suggestion}")
                
                with tab2:
                    st.write("**Raw OCR Output:**")
                    st.code(result['raw_text'])
                    
                    if st.session_state.ground_truth:
                        st.write("**Ground Truth Comparison:**")
                        st.code(st.session_state.ground_truth)
                        
                        # Simple accuracy calculation
                        def calculate_similarity(text1, text2):
                            text1, text2 = text1.lower(), text2.lower()
                            common = sum(1 for a, b in zip(text1, text2) if a == b)
                            return common / max(len(text1), len(text2))
                        
                        similarity = calculate_similarity(result['improved_text'], st.session_state.ground_truth)
                        st.metric("Text Similarity", f"{similarity:.2%}")
                
                if show_preprocessing and result['processed_img'] is not None:
                    st.subheader("🖼️ Image Processing")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    ax1.imshow(np.array(st.session_state.current_image))
                    ax1.set_title("Original")
                    ax1.axis('off')
                    ax2.imshow(result['processed_img'], cmap='gray')
                    ax2.set_title("Processed")
                    ax2.axis('off')
                    st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload an image or select a sample first")

# Footer
st.markdown("""
---
### ℹ️ About This App
- **OCR Engine**: EasyOCR
- **Text Improvement**: Basic spelling and grammar suggestions
- **Preprocessing**: Adaptive thresholding + denoising
""")