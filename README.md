# Generative-Data-Augmentation-for-Small-Datasets
# GenAI Data Augmentation Toolkit

🚀 A powerful **Streamlit application** for generating synthetic data across **tabular, text, and image** domains. This toolkit helps researchers, students, and startups overcome the challenge of limited datasets by creating realistic augmented samples using **Generative AI** and classical techniques.

---
Live Demo Link: https://generative-data-augmentation-for-small-datasets-19.streamlit.app/
## ✨ Features

### 📊 Tabular Data
- Generate synthetic rows using **SDV CTGAN** and **GaussianCopula**.
- Class balancing: ensure equal representation of all categories.
- Validation plots: compare distributions of real vs synthetic data.
- Export results as **CSV**.

### 📝 Text Data
- Generate paraphrases using a **cascading strategy**:
  - **PEGASUS paraphrasing** (if installed)
  - **Back-translation** (English → German → English)
  - **WordNet synonym replacement** (as fallback)
- Export results as **CSV**.

### 🖼️ Image Data
- Apply augmentations: random crop, flip, rotation, and color jitter.
- Optional: generate new images from prompts with **Stable Diffusion**.
- Export results as **ZIP**.

---

## 📦 Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/genai-data-augmentation.git
cd genai-data-augmentation
pip install -r requirements.txt
```

Download required NLTK resources:
```bash
python -m nltk.downloader wordnet omw-1.4
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501/`.

---

## 🧪 Example Workflows

### Tabular
1. Upload a **CSV file**.
2. Select target column for balancing (optional).
3. Generate synthetic rows and preview distribution plots.

### Text
1. Paste text lines (one per line).
2. Choose how many paraphrases per line.
3. Generate augmented text and download results.

### Images
1. Upload **images (JPG/PNG)**.
2. Select number of augmented variants.
3. (Optional) Provide a text prompt to generate new images via Stable Diffusion.

---

## 📌 Notes
- Heavy models (PEGASUS, MarianMT, Stable Diffusion) are **optional** – the app will fall back to lighter methods if they are not installed.
- GPU acceleration is recommended for Stable Diffusion.
- Use virtual environments to avoid dependency conflicts.

---

## 🗂️ Project Structure
```
├── app.py             # Main Streamlit app
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```

---

## 📜 License
MIT License – Free to use and modify for personal or commercial projects.

---

## 🙌 Acknowledgements
- [SDV](https://sdv.dev/) for tabular data synthesis
- [Hugging Face Transformers](https://huggingface.co/transformers/) for text models
- [TorchVision](https://pytorch.org/vision/stable/index.html) for image transforms
- [Diffusers](https://huggingface.co/docs/diffusers) for Stable Diffusion

---

💡 *With this toolkit, you can supercharge your small datasets and unlock better model performance!*
