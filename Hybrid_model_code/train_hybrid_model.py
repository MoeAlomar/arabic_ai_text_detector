import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix

LINGUISTIC_FILE = "features_gemini_vs_human_augmented.csv"
RAW_DATA_FILE = "merged_dataset_clean2.csv" 
COL_HUMAN = "human_collected_dataset"
COL_GEMINI = "gemini_rephrased_v2_5"

print("Starting hybrid model training (Farasa + N-grams)...")

if not os.path.exists(LINGUISTIC_FILE):
    print(f"Error: linguistic features file '{LINGUISTIC_FILE}' not found. Run the previous pipeline first.")
    exit()

df_features = pd.read_csv(LINGUISTIC_FILE)
df_features.dropna(inplace=True)
X_linguistic = df_features.drop(columns=['label'])
y = df_features['label']
print(f"Loaded linguistic features: {len(X_linguistic)} samples.")

try:
    df_raw = pd.read_csv(RAW_DATA_FILE)
    df_raw.columns = df_raw.columns.str.strip()
    
    df_human = pd.DataFrame({'text': df_raw[COL_HUMAN]})
    df_ai = pd.DataFrame({'text': df_raw[COL_GEMINI]})
    df_text = pd.concat([df_human, df_ai], ignore_index=True)
    
    min_len = min(len(df_features), len(df_text))
    df_text = df_text.iloc[:min_len]
    X_linguistic = X_linguistic.iloc[:min_len]
    y = y.iloc[:min_len]
    
    X_text = df_text['text'].astype(str)
    print(f"Loaded raw text: {len(X_text)} samples (synced).")

except Exception as e:
    print(f"Error loading raw text data: {e}")
    exit()

print("Generating N-gram features (TF-IDF character n-grams)...")

tfidf = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=20000, 
    min_df=5               
)
X_ngrams = tfidf.fit_transform(X_text)
print(f"N-gram features shape: {X_ngrams.shape}")

X_linguistic_sparse = csr_matrix(X_linguistic.values)

X_hybrid = hstack([X_ngrams, X_linguistic_sparse])

print(f"Hybrid dataset shape: {X_hybrid.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_hybrid, y, test_size=0.2, random_state=42, stratify=y
)

print("Training XGBoost hybrid model...")

model = XGBClassifier(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"Final hybrid model accuracy: {accuracy * 100:.2f}%")
print("="*50)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
