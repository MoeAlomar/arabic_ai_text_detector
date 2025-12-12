import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json

# --- Configuration ---
INPUT_FEATURES_FILE = "features_gemini_vs_human_augmented.csv"
MODEL_SAVE_PATH = "ai_detector_model.json"
FEATURE_NAMES_PATH = "feature_names.json"

print("🚀 Starting FINAL Model Training and EXPORT...")

# 1. Load Data
if not os.path.exists(INPUT_FEATURES_FILE):
    print(f"❌ Error: '{INPUT_FEATURES_FILE}' not found.")
    exit()

df = pd.read_csv(INPUT_FEATURES_FILE)
df.dropna(inplace=True)

X = df.drop(columns=['label'])
y = df['label']

# 2. Split Data (Using the same split as the Grid Search)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# 3. Train XGBoost Model with FINAL WINNING PARAMETERS
# الإعدادات الفائزة من آخر محاولة (84.18%)
WINNING_PARAMS = {
    'learning_rate': 0.03, 
    'max_depth': 3, 
    'min_child_weight': 5, 
    'n_estimators': 1200,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

model = XGBClassifier(
    **WINNING_PARAMS,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

print("🤖 Training Model with FINAL Optimized Parameters...")
model.fit(X_train, y_train)

# 4. EXPORT / SAVE THE MODEL AND FEATURES
# أ. حفظ المودل بتنسيق JSON
model.save_model(MODEL_SAVE_PATH)
print(f"✅ المودل حفظ بنجاح: {MODEL_SAVE_PATH}")

# ب. حفظ أسماء الخصائص (ضروري لترتيب الإدخال في Gradio)
feature_names = X.columns.tolist()
with open(FEATURE_NAMES_PATH, 'w', encoding='utf-8') as f:
    json.dump(feature_names, f, ensure_ascii=False, indent=4)
print(f"✅ أسماء الخصائص حفظت بنجاح: {FEATURE_NAMES_PATH}")

# 5. Final Evaluation (للتأكد)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*40)
print(f"🏆 Final Exported Model Accuracy: {accuracy * 100:.2f}%")
print("="*40)