
````
# 🧠 Arabic AI Text Detector | كاشف النصوص العربية

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)

A robust machine learning application designed to distinguish between **Human-written** and **AI-generated** text in Arabic. This project features a dual-model approach, hosting both a lightweight **Hybrid XGBoost** model and a powerful **Fine-Tuned AraBERT** deep learning model in a single interface.

##  Live Demo
Try the app directly on Hugging Face Spaces:
**[Arabic AI Text Detectors Space](https://huggingface.co/spaces/MoeAlomar/Arabic_ai_text_detectors)**

---

## Models Overview

This repository implements two distinct approaches to AI detection:

### 1. Hybrid XGBoost Model (Statistical & Linguistic)
This model combines traditional text analysis with linguistic feature engineering. It is fast and interpretable.
* **Input Processing:** TF-IDF Vectorization.
* **Linguistic Features:** Extracts 13 stylistic features using `Spacy`, including:
    * Part-of-Speech Ratios (Nouns, Verbs, Adjectives).
    * Average Sentence Length & Word Length.
    * Type-Token Ratio (TTR) for vocabulary richness.
    * Punctuation usage.
* **Classifier:** XGBoost (Extreme Gradient Boosting).

### 2. Fine-Tuned AraBERT (Deep Learning)
A Transformer-based model optimized for understanding semantic context in Arabic.
* **Base Model:** `aubmindlab/bert-base-arabertv02`
* **Preprocessing:** Uses `ArabertPreprocessor` for text normalization.
* **Training:** Fine-tuned on a specialized dataset of human vs. AI Arabic text.

---

## Project Structure

```text
arabic_ai_text_detector/
│
├── app.py                     # Main application file (Gradio UI)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
└── Models/                    # Trained Model Files
    ├── Hybrid_XGBoost_model/
    │   ├── hybrid_xgb.pkl               # The XGBoost Classifier (Joblib)
    │   ├── tfidf_vectorizer.pkl         # TF-IDF Vectorizer
    │   └── linguistic_feature_columns.pkl # Feature definitions
    │
    └── Fine_tuned_model/
        ├── config.json
        ├── pytorch_model.bin            # The Fine-Tuned BERT weights
        ├── tokenizer.json
        └── vocab.txt
````

-----

## 📦 Requirements

  * `gradio`
  * `xgboost`
  * `scikit-learn`
  * `pandas`
  * `numpy`
  * `spacy`
  * `torch`
  * `transformers`
  * `arabert`
  * `joblib`

-----

## 📝 Usage

1.  **Select a Tab:** Choose between the **Hybrid Model** (Statistical) or **AraBERT Model** (Deep Learning).
2.  **Input Text:** Paste the Arabic text you want to analyze into the text box.
3.  **Analyze:** Click the button to run the prediction.
4.  **View Results:** The model will display the probability of the text being "Human-Written" vs. "AI-Generated".


```
```
