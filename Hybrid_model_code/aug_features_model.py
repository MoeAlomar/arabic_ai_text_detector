import pandas as pd
from farasa.pos import FarasaPOSTagger
from tqdm import tqdm
import os
import re
import urllib3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

INPUT_FILE = "merged_dataset_clean2.csv"
FEATURES_FILE = "features_gemini_vs_human_augmented.csv"

COL_HUMAN = "human_collected_dataset"
COL_GEMINI = "gemini_rephrased_v2_5"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print(f"Starting final pipeline: {COL_HUMAN} vs {COL_GEMINI}")

if os.path.exists(FEATURES_FILE):
    print(f"Feature file '{FEATURES_FILE}' found. Loading features.")
    final_dataset = pd.read_csv(FEATURES_FILE)
else:
    print("Feature file not found. Extracting features from raw data.")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: input file '{INPUT_FILE}' not found.")
        exit()

    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} rows.")
        df.columns = df.columns.str.strip()
        if COL_HUMAN not in df.columns or COL_GEMINI not in df.columns:
            print("Required columns not found.")
            print(f"Available columns: {df.columns.tolist()}")
            exit()
    except Exception as e:
        print(f"Failed to read input file: {e}")
        exit()

    df = df[~df[COL_GEMINI].astype(str).str.contains("SKIPPED|ERROR", na=False)]
    df.dropna(subset=[COL_HUMAN, COL_GEMINI], inplace=True)

    df_human = pd.DataFrame({'text': df[COL_HUMAN], 'label': 0})
    df_gemini = pd.DataFrame({'text': df[COL_GEMINI], 'label': 1})
    df_final = pd.concat([df_human, df_gemini], ignore_index=True)

    df_final['text'] = df_final['text'].astype(str)
    df_final = df_final[df_final['text'].str.strip().str.len() > 10]

    print(f"Total valid samples: {len(df_final)}")

    try:
        pos_tagger = FarasaPOSTagger(interactive=True)
    except Exception as e:
        print(f"Failed to initialize Farasa: {e}")
        exit()

    def extract_features(text):
        features = {
            'NOUN_ratio': 0.0, 'VERB_ratio': 0.0, 'PART_ratio': 0.0, 'ADJ_ratio': 0.0,
            'NUM_ratio': 0.0, 'PRON_ratio': 0.0, 'DET_ratio': 0.0, 'PUNC_ratio': 0.0,
            'avg_word_len': 0.0, 'word_count': 0,
            'TTR_ratio': 0.0,
            'avg_sentence_len': 0.0,
            'UNKNOWN_ratio': 0.0
        }

        if not text:
            return features

        try:
            sentences = re.split(r'[.؟!]', text)
            sentences = [s for s in sentences if s.strip()]

            tagged_text = pos_tagger.tag(text)
            if not tagged_text:
                return features

            tokens = tagged_text.split()
            total_tokens = len(tokens)
            features['word_count'] = total_tokens
            if total_tokens == 0:
                return features

            unknown_count = 0
            clean_words = []

            for token in tokens:
                if '/' not in token:
                    unknown_count += 1
                    continue

                word, tag = token.rsplit('/', 1)
                clean_words.append(word)

                if tag.startswith('S') or tag in ['NOUN', 'FOREIGN']:
                    features['NOUN_ratio'] += 1
                elif tag.startswith('V'):
                    features['VERB_ratio'] += 1
                elif tag.startswith('PART') or tag in ['CONJ', 'PREP', 'PRON', 'H']:
                    features['PART_ratio'] += 1
                elif tag.startswith('ADJ'):
                    features['ADJ_ratio'] += 1
                elif tag.startswith('NUM') or tag == 'NSUFF':
                    features['NUM_ratio'] += 1
                elif tag.startswith('PRON'):
                    features['PRON_ratio'] += 1
                elif tag.startswith('DET'):
                    features['DET_ratio'] += 1
                elif tag == 'PUNC':
                    features['PUNC_ratio'] += 1

            if clean_words:
                features['avg_word_len'] = sum(len(w) for w in clean_words) / len(clean_words)
                features['TTR_ratio'] = len(set(clean_words)) / len(clean_words)

            if sentences:
                features['avg_sentence_len'] = sum(len(s.split()) for s in sentences) / len(sentences)

            features['UNKNOWN_ratio'] = unknown_count / total_tokens

            for key in features:
                if key not in ['word_count', 'avg_word_len', 'TTR_ratio', 'avg_sentence_len', 'UNKNOWN_ratio']:
                    features[key] /= total_tokens

        except Exception:
            pass

        return features

    print("Extracting features...")
    tqdm.pandas()
    features_df = df_final['text'].progress_apply(extract_features).apply(pd.Series)
    final_dataset = pd.concat([features_df, df_final['label']], axis=1)
    final_dataset = final_dataset[final_dataset['word_count'] > 0]
    final_dataset.to_csv(FEATURES_FILE, index=False)
    print(f"Features saved to '{FEATURES_FILE}'")

print("Starting model training and optimization")

final_dataset.dropna(inplace=True)
X = final_dataset.drop(columns=['label'])
y = final_dataset['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

param_grid = {
    'n_estimators': [900, 1200, 1500],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}

xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("Training model...")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 40)
print(f"Final accuracy: {accuracy * 100:.2f}%")
print("=" * 40)

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

feature_importance = best_model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame(
    list(feature_importance.items()),
    columns=['feature', 'score']
).sort_values(by='score', ascending=False)

print("Top features:")
print(importance_df.head(12))
