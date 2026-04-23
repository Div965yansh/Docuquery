# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import io

# PDF / DOCX
import pdfplumber
from docx import Document

# OCR (Images)
import pytesseract
from PIL import Image
import cv2
import numpy as np

# -------------------------------------------------

app = Flask(__name__)
CORS(app)

MODEL_DIR = "agnews-distilbert"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# ------------------ Model loading ------------------

device_is_cuda = torch.cuda.is_available()
device_id = 0 if device_is_cuda else -1

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

if device_is_cuda:
    model.to("cuda")

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device_id
)

# ------------------ OCR config ------------------

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------ Helpers ------------------

def chunk_text(text, max_chars=1200):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def extract_text_from_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    return pytesseract.image_to_string(gray)


def extract_text(uploaded_file):
    filename = uploaded_file.filename.lower()
    file_bytes = uploaded_file.read()

    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n{page_text}"
        return text

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        return extract_text_from_image(file_bytes)

    return None


# ------------------ Core classification ------------------

def classify_text(text):
    chunks = chunk_text(text)

    if not chunks:
        return {
            "final_category": "Unknown",
            "overall_confidence": 0,
            "document_type": "Empty",
            "chunks_processed": 0,
            "category_distribution": [],
            "chunk_analysis": []
        }

    scores = {label: 0.0 for label in LABELS}
    chunk_analysis = []

    for idx, chunk in enumerate(chunks):
        out = classifier(chunk)[0]
        label_id = int(out["label"].split("_")[-1])
        label = LABELS[label_id]
        confidence = float(out["score"])

        scores[label] += confidence

        chunk_analysis.append({
            "chunk_number": idx + 1,
            "predicted_category": label,
            "confidence": round(confidence, 3)
        })

    total_score = sum(scores.values())

    category_distribution = [
        {
            "label": label,
            "confidence": round(scores[label] / total_score, 3)
        }
        for label in LABELS
    ]

    category_distribution.sort(key=lambda x: x["confidence"], reverse=True)

    primary = category_distribution[0]

    document_type = (
        "Single-topic" if primary["confidence"] >= 0.65 else "Mixed"
    )

    return {
        "final_category": primary["label"],
        "overall_confidence": round(primary["confidence"], 3),
        "document_type": document_type,
        "chunks_processed": len(chunks),
        "category_distribution": category_distribution,
        "chunk_analysis": chunk_analysis
    }


# ------------------ Routes ------------------

@app.route("/predict", methods=["POST"])
def predict_text():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    result = classify_text(text)
    result["text_preview"] = text[:1000]

    return jsonify(result)


@app.route("/predict_file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    text = extract_text(uploaded_file)
    if not text or not text.strip():
        return jsonify({"error": "Could not extract text"}), 400

    result = classify_text(text)
    result["text_preview"] = text[:1000]

    return jsonify(result)


# ------------------ Run ------------------

if __name__ == "__main__":
    app.run(port=5000, debug=True)
