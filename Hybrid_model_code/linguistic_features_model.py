import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# --- Configuration ---
INPUT_FEATURES_FILE = "features_gemini_vs_human.csv" 

print("🚀 Starting Model Training...")

# 1. Load Data
if not os.path.exists(INPUT_FEATURES_FILE):
    print(f"❌ Error: '{INPUT_FEATURES_FILE}' not found. Run extract_features.py first.")
    exit()

df = pd.read_csv(INPUT_FEATURES_FILE)
print(f"📥 Loaded {len(df)} samples.")

# 2. Prepare Data
# حذف الصفوف التي قد تحتوي على قيم NaN إذا لم تكن Farasa قد ملأتها بالكامل
df.dropna(inplace=True)
print(f"📊 Samples after final dropna: {len(df)}")


X = df.drop(columns=['label'])
y = df['label']

# تقسيم البيانات (تدريب 80% واختبار 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 3. Train XGBoost Model
# نستخدم معلمات (Parameters) قوية ومناسبة لبيانات الـ Stylometry
model = XGBClassifier(
    n_estimators=300,        # عدد الشجيرات
    learning_rate=0.05,      # معدل التعلم
    max_depth=6,             # عمق الشجرة الأقصى
    subsample=0.8,           
    colsample_bytree=0.8,    
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

print("🤖 Training XGBoost Model...")
# قد تستغرق هذه الخطوة بضع ثواني أو دقيقة واحدة
model.fit(X_train, y_train)

# 4. Evaluation
print("✅ Training Complete. Evaluating...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*40)
print(f"🏆 Final Model Accuracy: {accuracy * 100:.2f}%")
print("="*40)

print("\n📋 Classification Report:")
# هذا التقرير يوضح أداء النموذج على كل فئة (Human vs AI)
print(classification_report(y_test, y_pred, target_names=['Human (0)', 'AI (1)']))

# 5. Feature Importance (Explainability)
# هذا يوضح أي الخصائص اللغوية كانت أهم في كشف الـ AI
print("\n🔍 Top 10 Most Important Linguistic Features:")
feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

importance_df = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
print(importance_df.head(10))
