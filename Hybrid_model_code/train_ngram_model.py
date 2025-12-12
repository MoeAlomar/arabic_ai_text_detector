import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

INPUT_FILE = "merged_dataset_clean2.csv"
COL_HUMAN = "human_collected_dataset"
COL_GEMINI = "gemini_rephrased_v2_5"

print("Starting N-grams model training...")

if not os.path.exists(INPUT_FILE):
    print(f"Error: file '{INPUT_FILE}' not found.")
    exit()

try:
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()
    
    if COL_HUMAN not in df.columns or COL_GEMINI not in df.columns:
        print("Columns not found.")
        exit()

    df.dropna(subset=[COL_HUMAN, COL_GEMINI], inplace=True)
    df = df[~df[COL_GEMINI].astype(str).str.contains("SKIPPED|ERROR", na=False)]

    print(f"Loaded {len(df)} rows.")

    df_human = pd.DataFrame({'text': df[COL_HUMAN], 'label': 0})
    df_gemini = pd.DataFrame({'text': df[COL_GEMINI], 'label': 1})
    df_final = pd.concat([df_human, df_gemini], ignore_index=True)
    
    df_final['text'] = df_final['text'].astype(str)
    df_final = df_final[df_final['text'].str.strip().str.len() > 10]

    print(f"Training samples: {len(df_final)}")

except Exception as e:
    print(f"Error: {e}")
    exit()

print("Vectorizing text (character N-grams)...")

tfidf = TfidfVectorizer(
    analyzer='char', 
    ngram_range=(2, 5),
    max_features=20000,
    min_df=5
)

X = tfidf.fit_transform(df_final['text'])
y = df_final['label']

print(f"Vectorization complete. Features shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*40)
print(f"N-grams model accuracy: {accuracy * 100:.2f}%")
print("="*40)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

print("\nTop 15 N-grams predicting AI vs Human:")
feature_names = tfidf.get_feature_names_out()
coefs = model.coef_[0]

top_positive_indices = np.argsort(coefs)[-15:][::-1]
top_negative_indices = np.argsort(coefs)[:15]

print("\nStrongest indicators for AI (Gemini):")
for idx in top_positive_indices:
    print(f"   '{feature_names[idx]}' (Score: {coefs[idx]:.2f})")

print("\nStrongest indicators for Human:")
for idx in top_negative_indices:
    print(f"   '{feature_names[idx]}' (Score: {coefs[idx]:.2f})")
