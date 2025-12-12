import gradio as gr
import pickle
import pandas as pd
import numpy as np
import spacy
import torch
import os
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor


# 📂 FOLDER CONFIGURATION

# Make sure your folders are named exactly this in your GitHub repo!
PATH_XGB_ENGLISH = "Models/Hybrid_XGBoost_model/"
PATH_BERT_ARABIC = "Models/Fine_tuned_model/"

print("⏳ Starting AI Detection App...")

# LOAD Hybrid MODEL

try:
    # Check/Download Spacy Model for linguistic features
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading Spacy model...")
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Load the 3 Pickle files
    print(f"Loading XGBoost from {PATH_XGB_ENGLISH}...")
    with open(os.path.join(PATH_XGB_ENGLISH, "hybrid_xgb.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(PATH_XGB_ENGLISH, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(os.path.join(PATH_XGB_ENGLISH, "linguistic_feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)

    print("✅ Hybrid Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading Hybrid Model: {e}")
    xgb_model = None

# LOAD ARABIC MODEL (BERT)

try:
    print(f"Loading BERT from {PATH_BERT_ARABIC}...")
    device = "cpu"

    # Load Tokenizer & Model
    # This expects 'pytorch_model.bin' or 'model.safetensors' to be in the folder!
    tokenizer_bert = AutoTokenizer.from_pretrained(PATH_BERT_ARABIC)
    model_bert = AutoModelForSequenceClassification.from_pretrained(
        PATH_BERT_ARABIC).to(device)

    # Initialize Arabert Preprocessor (downloads config automatically)
    arabert_prep = ArabertPreprocessor(
        model_name="aubmindlab/bert-base-arabertv02")

    print("✅ Fine tuned Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading Model: {e}")
    print("⚠️  Did you forget to upload the 'pytorch_model.bin' or 'model.safetensors' file?")
    model_bert = None

# ==========================================
# 🧠 FEATURE EXTRACTION & PREDICTION
# ==========================================


def get_english_features(text):
    """Extracts linguistic features using Spacy (Noun ratio, etc.)"""
    doc = nlp(text)
    word_count = len([t for t in doc if not t.is_punct])
    if word_count == 0:
        word_count = 1

    pos = doc.count_by(spacy.attrs.POS)

    features = {
        "NOUN_ratio": pos.get(spacy.symbols.NOUN, 0) / word_count,
        "VERB_ratio": pos.get(spacy.symbols.VERB, 0) / word_count,
        "PART_ratio": pos.get(spacy.symbols.PART, 0) / word_count,
        "ADJ_ratio":  pos.get(spacy.symbols.ADJ, 0) / word_count,
        "NUM_ratio":  pos.get(spacy.symbols.NUM, 0) / word_count,
        "PRON_ratio": pos.get(spacy.symbols.PRON, 0) / word_count,
        "DET_ratio":  pos.get(spacy.symbols.DET, 0) / word_count,
        "PUNC_ratio": pos.get(spacy.symbols.PUNCT, 0) / word_count,
        "avg_word_len": sum(len(t.text) for t in doc) / len(doc) if len(doc) > 0 else 0,
        "word_count": word_count,
        "TTR_ratio": len(set([t.text.lower() for t in doc])) / word_count,
        "avg_sentence_len": word_count / len(list(doc.sents)) if len(list(doc.sents)) > 0 else 0,
        "UNKNOWN_ratio": 0
    }
    # Return as DataFrame with columns in the exact order the model expects
    return pd.DataFrame([features])[feature_columns]


def predict_english(text):
    if xgb_model is None:
        return {"Error": "XGBoost model files missing."}
    if not text.strip():
        return "Please enter text."

    try:
        # 1. Transform Text (TF-IDF)
        tfidf_data = tfidf_vectorizer.transform([text])
        # 2. Get Linguistic Features
        ling_data = get_english_features(text)
        # 3. Combine
        full_data = hstack([tfidf_data, ling_data])
        # 4. Predict
        probs = xgb_model.predict_proba(full_data)[0]

        # Result: Class 0 is Human, Class 1 is AI (Verify your training labels if swapped!)
        return {"👤 Human Written": float(probs[0]), "🤖 AI Generated": float(probs[1])}
    except Exception as e:
        return f"Prediction Error: {str(e)}"


def predict_arabic(text):
    if model_bert is None:
        return {"Error": "Arabic model files missing (check pytorch_model.bin)."}
    if not text.strip():
        return "الرجاء إدخال نص."

    try:
        # 1. Preprocess
        prep_text = arabert_prep.preprocess(text)
        # 2. Tokenize
        inputs = tokenizer_bert(
            prep_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(device)

        # 3. Predict
        with torch.no_grad():
            outputs = model_bert(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        # Result
        prob_human = float(probs[0][0])
        prob_ai = float(probs[0][1])

        return {"👤 نص بشري": prob_human, "🤖 ذكاء اصطناعي": prob_ai}
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 🎨 GRADIO UI (TABS)
# ==========================================


with gr.Blocks(title="AI Text Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ Global AI Text Detector")
    gr.Markdown(
        "Detect if text was written by a Human or AI in English and Arabic.")

    with gr.Tabs():

        # --- TAB 1: Hybrid ---
        with gr.TabItem(" Hybrid Model Detector"):
            gr.Markdown("### Hybrid XGBoost Model")
            gr.Markdown(
                "This model analyzes vocabulary (TF-IDF) and grammar stats (Noun/Verb ratios).")

            with gr.Row():
                eng_input = gr.Textbox(
                    lines=5, label="English Text", placeholder="Paste article or essay here...")
                eng_output = gr.Label(label="Probability")

            eng_btn = gr.Button("Analyze English Text", variant="primary")
            eng_btn.click(predict_english, inputs=eng_input,
                          outputs=eng_output)

        # --- TAB 2: ARABIC ---
        with gr.TabItem("AraBERT Detector"):
            gr.Markdown("### Fine-Tuned AiBERT Model")
            gr.Markdown("نموذج ذكاء اصطناعي متخصص في كشف النصوص العربية")

            with gr.Row():
                ar_input = gr.Textbox(
                    lines=5, label="النص العربي", placeholder="ضع النص هنا...", rtl=True)
                ar_output = gr.Label(label="النتيجة")

            ar_btn = gr.Button("تحليل النص", variant="primary")
            ar_btn.click(predict_arabic, inputs=ar_input, outputs=ar_output)

if __name__ == "__main__":
    demo.launch()
