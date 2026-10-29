# 📝 Multilingual OCR Web Application

A **deep learning-powered OCR web app** that extracts text from images across multiple Indian languages using state-of-the-art models.

🚀 **Live Demo:**
👉 https://huggingface.co/spaces/varshaa-1616/multilingual_ocr

---

## 📌 Overview

This project is a **multilingual Optical Character Recognition (OCR) system** deployed as a web application.
It allows users to upload or paste images and extract text in different languages using specialized OCR pipelines.

The system combines:

* 🧠 Deep learning models
* 🖼️ Image preprocessing
* 🌐 Interactive web interface (Gradio)

---

## ✨ Features

* 📤 Upload or paste images directly (clipboard supported)
* 🌐 Multi-language OCR support:

  * Urdu
  * Hindi
  * Kannada
* 🧪 Built-in sample images for quick demo
* ⚡ Fast inference with optimized preprocessing
* 🧑‍💻 Clean and interactive UI
* ☁️ Deployed using Docker on Hugging Face Spaces

---

## 🧠 Models Used

### 🔹 Urdu OCR

* **Framework:** EasyOCR
* **Features:**

  * Handles Urdu script + mixed English text
  * Custom preprocessing (denoising + thresholding)
  * Text normalization (removes diacritics and noise)

---

### 🔹 Hindi & Kannada OCR

* **Framework:** IndicPhotoOCR
* **Architecture includes:**

  * Vision Transformer (ViT) for script identification
  * Text detection using TextBPN++
  * Recognition pipeline optimized for Indic scripts

---

## ⚙️ Tech Stack

* **Frontend/UI:** Gradio
* **Backend:** Python
* **Deep Learning:** PyTorch
* **Libraries:**

  * EasyOCR
  * Transformers (Hugging Face)
  * OpenCV
  * NumPy
  * IndicPhotoOCR
* **Deployment:** Docker + Hugging Face Spaces

---

## 📂 Project Structure

```bash
multilingual-ocr-app/
│
├── app.py                 # Main Gradio application
├── requirements.txt       # Dependencies
├── Dockerfile             # Deployment container
├── README.md              # Documentation
│
├── sample_images/         # Demo images
│   ├── urdu/
│   ├── hindi/
│   └── kannada/
│
└── IndicPhotoOCR/         # OCR model repository
```

---

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/multilingual-ocr-app.git
cd multilingual-ocr-app
```

---

### 2. Create environment

```bash
conda create -n ocrapp python=3.10 -y
conda activate ocrapp
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the app

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:7860
```

---

## 🐳 Docker Deployment (Optional)

```bash
docker build -t ocr-app .
docker run -p 7860:7860 ocr-app
```

---

## 🌐 Deployment

This app is deployed using:

* Docker container
* Hugging Face Spaces

👉 Live link:
https://huggingface.co/spaces/varshaa-1616/multilingual_ocr

---

## 📊 Workflow

1. User uploads/pastes image
2. Selects language
3. Image preprocessing applied
4. Routed to correct OCR model
5. Text extracted and displayed

---

## ⚠️ Challenges & Solutions

| Challenge                            | Solution                    |
| ------------------------------------ | --------------------------- |
| IndicPhotoOCR installation issues    | Used local repo + Docker    |
| PyTorch 2.6 checkpoint loading error | Downgraded to PyTorch 2.5   |
| Dependency conflicts                 | Cleaned and pinned versions |
| Multi-language handling              | Dynamic routing logic       |

---

## 🔮 Future Improvements

* 📦 Batch OCR (multiple images)
* 📊 Confidence scores
* 🧾 Download extracted text
* 📍 Bounding box visualization
* 🌍 More language support

---

## 👩‍💻 Mentor

**Panigrahi Srikanth**

---

## 👩‍💻 Author

**Srivarsha C**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---
