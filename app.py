import os
import re
import cv2
import glob
import uuid
import torch
import easyocr
import tempfile
import numpy as np
from PIL import Image
import gradio as gr

# -------------------------
# IndicPhotoOCR
# -------------------------
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDICOCR_REPO = os.path.join(BASE_DIR, "IndicPhotoOCR")
sys.path.insert(0, INDICOCR_REPO)

from IndicPhotoOCR.ocr import OCR

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SAMPLE_DIR = "sample_images"

# =========================
# LOAD MODELS ONCE
# =========================
urdu_reader = easyocr.Reader(['ur', 'en'], gpu=torch.cuda.is_available(), verbose=False)
indicocr_system = OCR(device=DEVICE, verbose=False)

# =========================
# URDU NORMALIZATION
# =========================
urdu_diacritics = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED]')
tatweel = "\u0640"

def normalize_urdu_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("\ufeff", "").replace("\u200c", "").replace("\u200d", "")
    s = s.replace(tatweel, "")
    s = urdu_diacritics.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================
# IMAGE HELPERS
# =========================
def ensure_rgb(np_img):
    if np_img is None:
        return None
    if len(np_img.shape) == 2:
        return cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    if np_img.shape[2] == 4:
        return cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
    return np_img

def save_temp_image(np_img):
    np_img = ensure_rgb(np_img)
    if np_img is None:
        return None
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, f"ocr_{uuid.uuid4().hex}.png")
    Image.fromarray(np_img).save(path)
    return path

# =========================
# URDU PREPROCESS
# =========================
def preprocess_for_urdu(np_img):
    np_img = ensure_rgb(np_img)
    if np_img is None:
        return None
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thr

# =========================
# OCR FUNCTIONS
# =========================
def infer_urdu(np_img):
    img = preprocess_for_urdu(np_img)
    if img is None:
        return "IMAGE NOT FOUND"
    result = urdu_reader.readtext(img, detail=0, paragraph=True)
    if isinstance(result, list):
        text = " ".join([str(x) for x in result])
    else:
        text = str(result)
    return normalize_urdu_text(text)

def infer_hindi(np_img):
    path = save_temp_image(np_img)
    if path is None:
        return "IMAGE NOT FOUND"
    result = indicocr_system.recognise(path, "hindi")
    return str(result).strip()

def infer_kannada(np_img):
    path = save_temp_image(np_img)
    if path is None:
        return "IMAGE NOT FOUND"
    result = indicocr_system.recognise(path, "kannada")
    return str(result).strip()

def run_ocr(image, language):
    if image is None:
        return "Please upload or paste an image."

    try:
        if language == "Urdu":
            return infer_urdu(image)
        elif language == "Hindi":
            return infer_hindi(image)
        elif language == "Kannada":
            return infer_kannada(image)
        else:
            return "Invalid language selected."
    except Exception as e:
        return f"Error during OCR: {str(e)}"

# =========================
# SAMPLE IMAGE LOGIC
# =========================
def get_sample_paths(language):
    folder_map = {
        "Urdu": os.path.join(BASE_SAMPLE_DIR, "urdu"),
        "Hindi": os.path.join(BASE_SAMPLE_DIR, "hindi"),
        "Kannada": os.path.join(BASE_SAMPLE_DIR, "kannada"),
    }
    folder = folder_map[language]
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files)[:5]
    return files

def update_gallery(language):
    sample_paths = get_sample_paths(language)
    if len(sample_paths) == 0:
        return gr.Gallery(value=[], label=f"Sample Images - {language}")
    gallery_items = [(p, os.path.basename(p)) for p in sample_paths]
    return gr.Gallery(value=gallery_items, label=f"Sample Images - {language}")

def gallery_to_input(evt: gr.SelectData, language):
    sample_paths = get_sample_paths(language)
    idx = evt.index
    if isinstance(idx, (tuple, list)):
        idx = idx[0]
    if idx is None or idx >= len(sample_paths):
        return None
    img = Image.open(sample_paths[idx]).convert("RGB")
    return np.array(img)

# =========================
# UI
# =========================
with gr.Blocks(title="Multilingual OCR Demo") as demo:
    gr.Markdown(
        """
        # Multilingual OCR Demo
        Upload or paste an image, choose the language, and extract text.

        **Supported languages:**
        - Urdu → EasyOCR
        - Hindi → IndicPhotoOCR
        - Kannada → IndicPhotoOCR
        """
    )

    with gr.Row():
        language = gr.Dropdown(
            choices=["Urdu", "Hindi", "Kannada"],
            value="Urdu",
            label="Choose Language"
        )

    with gr.Row():
        image_input = gr.Image(
            type="numpy",
            label="Upload or Paste Image",
            sources=["upload", "clipboard"]
        )

    with gr.Row():
        extract_btn = gr.Button("Extract Text", variant="primary")
        clear_btn = gr.Button("Clear")

    output_text = gr.Textbox(
        label="Extracted Text",
        lines=10,
        placeholder="OCR output will appear here..."
    )

    gr.Markdown("## Try sample images")
    sample_gallery = gr.Gallery(
        value=[(p, os.path.basename(p)) for p in get_sample_paths("Urdu")],
        label="Sample Images - Urdu",
        columns=5,
        height="auto",
        preview=True
    )

    language.change(
        fn=update_gallery,
        inputs=language,
        outputs=sample_gallery
    )

    sample_gallery.select(
        fn=gallery_to_input,
        inputs=language,
        outputs=image_input
    )

    extract_btn.click(
        fn=run_ocr,
        inputs=[image_input, language],
        outputs=output_text
    )

    clear_btn.click(
        fn=lambda: (None, ""),
        inputs=[],
        outputs=[image_input, output_text]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
