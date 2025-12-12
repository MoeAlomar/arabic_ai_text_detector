import gradio as gr
import pickle
import joblib
import pandas as pd
import spacy
import torch
import os
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor

# 📂 FOLDER CONFIGURATION
PATH_XGB_ENGLISH = "Models/Hybrid_XGBoost_model/"
PATH_BERT_ARABIC = "Models/Fine_tuned_model/"

print("⏳ Starting AI Detection App...")

# ==========================================
# ✅ LOAD SPACY (for linguistic features)
# ==========================================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading Spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ==========================================
# ✅ LOAD HYBRID MODEL (XGBoost + TFIDF + Feature Columns)
# ==========================================
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    print(f"Loading Hybrid XGBoost assets from {PATH_XGB_ENGLISH}...")

    xgb_path = os.path.join(PATH_XGB_ENGLISH, "hybrid_xgb.pkl")
    tfidf_path = os.path.join(PATH_XGB_ENGLISH, "tfidf_vectorizer.pkl")
    cols_path = os.path.join(PATH_XGB_ENGLISH, "linguistic_feature_columns.pkl")

    # XGBoost model is normal pickle
    xgb_model = load_pickle(xgb_path)

    # TFIDF is joblib in your case
    tfidf_vectorizer = joblib.load(tfidf_path)

    # Feature columns list is normal pickle
    feature_columns = load_pickle(cols_path)

    print("✅ Hybrid Model Loaded Successfully")

except Exception as e:
    print(f"❌ Error loading Hybrid Model: {e}")
    xgb_model = None
    tfidf_vectorizer = None
    feature_columns = None

# ==========================================
# ✅ LOAD ARABIC MODEL (Fine-tuned AraBERT)
# ==========================================
try:
    print(f"Loading BERT from {PATH_BERT_ARABIC}...")
    device = "cpu"

    tokenizer_bert = AutoTokenizer.from_pretrained(PATH_BERT_ARABIC)
    model_bert = AutoModelForSequenceClassification.from_pretrained(PATH_BERT_ARABIC).to(device)

    arabert_prep = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")

    print("✅ Fine tuned Model Loaded Successfully")

except Exception as e:
    print(f"❌ Error loading Model: {e}")
    print("⚠️  Did you forget to upload the 'pytorch_model.bin' or 'model.safetensors' file?")
    model_bert = None
    tokenizer_bert = None
    arabert_prep = None
    device = "cpu"

# ==========================================
# 🧠 FEATURE EXTRACTION & PREDICTION
# ==========================================
def get_features(text: str) -> pd.DataFrame:
    """Extract linguistic features using Spacy (POS ratios, etc.)"""
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
        "UNKNOWN_ratio": 0,
    }

    # IMPORTANT: keep the column order exactly as training time
    return pd.DataFrame([features])[feature_columns]


def predict_XGBoost(text: str):
    if xgb_model is None or tfidf_vectorizer is None or feature_columns is None:
        return {"Error": "Hybrid model files missing or failed to load."}

    if not text or not text.strip():
        return "Please enter text."

    try:
        tfidf_data = tfidf_vectorizer.transform([text])
        ling_data = get_features(text)
        full_data = hstack([tfidf_data, ling_data])

        probs = xgb_model.predict_proba(full_data)[0]
        return {"👤 نص بشري": float(probs[0]), "🤖 ذكاء اصطناعي": float(probs[1])}

    except Exception as e:
        return f"Prediction Error: {str(e)}"


def predict_arabic(text: str):
    if model_bert is None or tokenizer_bert is None or arabert_prep is None:
        return {"Error": "Arabic model files missing (check pytorch_model.bin / model.safetensors)."}

    if not text or not text.strip():
        return "الرجاء إدخال نص."

    try:
        prep_text = arabert_prep.preprocess(text)

        inputs = tokenizer_bert(
            prep_text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model_bert(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)

        prob_human = float(probs[0][0])
        prob_ai = float(probs[0][1])

        return {"👤 نص بشري": prob_human, "🤖 ذكاء اصطناعي": prob_ai}

    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 🎨 GRADIO UI (TABS)
# ==========================================
with gr.Blocks(title="AI Text Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Arabic AI Generated Text Detector")
    gr.Markdown("Detect if text was written by a Human or AI in Arabic.")

    with gr.Tabs():

        # --- TAB 1: Hybrid ---
        with gr.TabItem("Hybrid Model Detector"):
            gr.Markdown("### Hybrid XGBoost Model (Higher accuracy)")
            gr.Markdown("This model uses hybrid approaches to detect Arabic AI generated text.")

            with gr.Row():
                arr_input = gr.Textbox(lines=5, label="Text", placeholder="اكتب النص هنا...", rtl=True)
                arr_output = gr.Label(label="الإحتمالية")

            eng_btn = gr.Button("حلل النص", variant="primary")
            eng_btn.click(predict_XGBoost, inputs=arr_input, outputs=arr_output)

        # --- TAB 2: ARABIC ---
        with gr.TabItem("AraBERT Detector"):
            gr.Markdown("### Fine-Tuned AraBERT Model")
            gr.Markdown("Fine-tuned AraBERT model using our own dataset to detect Arabic AI generated text.")

            with gr.Row():
                ar_input = gr.Textbox(lines=5, label="النص العربي", placeholder="ضع النص هنا...", rtl=True)
                ar_output = gr.Label(label="النتيجة")

            ar_btn = gr.Button("حلل النص", variant="primary")
            ar_btn.click(predict_arabic, inputs=ar_input, outputs=ar_output)

if __name__ == "__main__":
    demo.launch()
