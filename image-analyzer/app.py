import streamlit as st
from PIL import Image
from nvidia_api import analyze_image_with_nvidia

# Page setup
st.set_page_config(page_title="Image Analyzer", layout="centered", page_icon="🦙")

# Custom CSS for dark/light mode + rounded layout
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
            color: #333;
        }
        .stApp {
            background-color: #f9f9f9;
            padding: 2rem;
        }
        @media (prefers-color-scheme: dark) {
            .stApp {
                background-color: #0f1117;
                color: white;
            }
        }
        .title-box {
            border-radius: 12px;
            background: linear-gradient(to right, #4c68d7, #6f4ebf);
            color: white;
            padding: 1rem;
            margin-bottom: 1rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title-box"><h2> Vision Image Analyzer</h2><p>Upload an image and let AI describe or analyze it.</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Drag & drop or browse", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    prompt = st.text_area(" Prompt: What should LLaMA look for?", placeholder="e.g. Describe this image, What objects are present, etc.")

    submit = st.button("Analyze Image ")

# Main content
if uploaded_file and prompt and submit:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image using LLaMA 4 Vision..."):
        result = analyze_image_with_nvidia(image, prompt)

    st.markdown("###  AI Analysis Result")
    st.success(result)

elif not uploaded_file:
    st.info(" Please upload an image from the sidebar to get started.")
elif uploaded_file and not prompt:
    st.warning("Please enter a prompt before analyzing.")
