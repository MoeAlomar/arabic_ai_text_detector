import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix # لدمج مصفوفات البيانات المتفرقة

# --- 1. الإعدادات ---
# ملف الخصائص اللغوية (الناتج من Farasa)
LINGUISTIC_FILE = "features_gemini_vs_human_augmented.csv"
# ملف النصوص الخام (لاستخراج N-Grams)
RAW_DATA_FILE = "merged_dataset_clean2.csv" 
COL_HUMAN = "human_collected_dataset"
COL_GEMINI = "gemini_rephrased_v2_5"

print("🚀 Starting Hybrid Model Training (Farasa + N-Grams)...")

# --- 2. تحميل وتجهيز البيانات ---

# أ. تحميل الخصائص اللغوية
if not os.path.exists(LINGUISTIC_FILE):
    print(f"❌ Error: Linguistic features file '{LINGUISTIC_FILE}' not found. Run previous pipeline first.")
    exit()

df_features = pd.read_csv(LINGUISTIC_FILE)
df_features.dropna(inplace=True)
X_linguistic = df_features.drop(columns=['label'])
y = df_features['label']
print(f"✅ Loaded Linguistic Features: {len(X_linguistic)} samples.")


# ب. تحميل النصوص الخام (لـ N-Grams)
try:
    df_raw = pd.read_csv(RAW_DATA_FILE)
    df_raw.columns = df_raw.columns.str.strip()
    
    # دمج النصوص الخام بنفس الترتيب
    df_human = pd.DataFrame({'text': df_raw[COL_HUMAN]})
    df_ai = pd.DataFrame({'text': df_raw[COL_GEMINI]})
    df_text = pd.concat([df_human, df_ai], ignore_index=True)
    
    # يجب أن تتطابق الأحجام
    min_len = min(len(df_features), len(df_text))
    df_text = df_text.iloc[:min_len]
    X_linguistic = X_linguistic.iloc[:min_len]
    y = y.iloc[:min_len]
    
    X_text = df_text['text'].astype(str)
    print(f"✅ Loaded Raw Text: {len(X_text)} samples (Synced).")

except Exception as e:
    print(f"❌ Error loading raw text data: {e}")
    exit()

# 3. استخراج خصائص N-Grams
print("⚙️  Generating N-Gram Features (TF-IDF Character N-grams)...")

tfidf = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=20000, 
    min_df=5               
)
X_ngrams = tfidf.fit_transform(X_text)
print(f"✅ N-Grams Features Shape: {X_ngrams.shape}")

# 4. دمج الخصائص (Hybrid Concatenation)
# نحول خصائص فراسة إلى مصفوفة متفرقة (Sparse Matrix) لدمجها مع TF-IDF
X_linguistic_sparse = csr_matrix(X_linguistic.values)

# الدمج الأفقي (الخصائص اللغوية + خصائص N-Grams)
X_hybrid = hstack([X_ngrams, X_linguistic_sparse])

print(f"✅ Hybrid Dataset Shape: {X_hybrid.shape}")

# 5. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X_hybrid, y, test_size=0.2, random_state=42, stratify=y
)

# 6. تدريب XGBoost (باستخدام أفضل المعلمات التي وجدتها سابقاً)
print("🤖 Training XGBoost Hybrid Model...")

model = XGBClassifier(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 7. التقييم
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"🏆 Final Hybrid Model Accuracy: {accuracy * 100:.2f}%")
print("="*50)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))