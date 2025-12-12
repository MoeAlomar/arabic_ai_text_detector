import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

INPUT_FEATURES_FILE = "features_gemini_vs_human.csv" 

print("Starting model training...")

if not os.path.exists(INPUT_FEATURES_FILE):
    print(f"Error: '{INPUT_FEATURES_FILE}' not found. Run extract_features.py first.")
    exit()

df = pd.read_csv(INPUT_FEATURES_FILE)
print(f"Loaded {len(df)} samples.")

df.dropna(inplace=True)
print(f"Samples after final dropna: {len(df)}")

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

print("Training XGBoost model...")
model.fit(X_train, y_train)

print("Training complete. Evaluating...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n" + "=" * 40)
print(f"Final model accuracy: {accuracy * 100:.2f}%")
print("=" * 40)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Human (0)', 'AI (1)']))

print("\nTop 10 most important linguistic features:")
feature_important = model.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

importance_df = pd.DataFrame(
    data=values,
    index=keys,
    columns=["score"]
).sort_valu_
