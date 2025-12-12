import pandas as pd
from farasa.pos import FarasaPOSTagger 
from tqdm import tqdm
import os
import sys

INPUT_FILE = "merged_dataset_clean2.csv"
OUTPUT_FILE = "features_gemini_vs_human.csv"

COL_HUMAN = "human_collected_dataset"
COL_GEMINI = "gemini_rephrased_v2_5"

print(f"Starting feature extraction: {COL_HUMAN} vs {COL_GEMINI}...")

if not os.path.exists(INPUT_FILE):
    print(f"Error: file '{INPUT_FILE}' not found.")
    exit()

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"File loaded: {len(df)} rows.")
except Exception as e:
    print(f"Failed to read file: {e}")
    exit()

if COL_HUMAN not in df.columns or COL_GEMINI not in df.columns:
    print(f"Required columns not found.\nAvailable: {df.columns.tolist()}")
    exit()

print("Ignoring other columns and focusing on Gemini only...")

df_human = pd.DataFrame({
    'text': df[COL_HUMAN],
    'label': 0
})

df_gemini = pd.DataFrame({
    'text': df[COL_GEMINI],
    'label': 1
})

df_final = pd.concat([df_human, df_gemini], ignore_index=True)

df_final.dropna(subset=['text'], inplace=True)
df_final['text'] = df_final['text'].astype(str)
df_final = df_final[df_final['text'].str.strip().str.len() > 5] 

print(f"Total samples for processing: {len(df_final)}")

print("Initializing Farasa POS Tagger...")
try:
    pos_tagger = FarasaPOSTagger(interactive=True)
    print("Farasa initialized successfully.")
except Exception as e:
    print(f"Failed to initialize Farasa: {e}")
    exit()

def extract_features(text):
    features = {
        'NOUN_ratio': 0.0, 'VERB_ratio': 0.0, 'PART_ratio': 0.0, 'ADJ_ratio': 0.0,
        'NUM_ratio': 0.0, 'PRON_ratio': 0.0, 'DET_ratio': 0.0, 'PUNC_ratio': 0.0,
        'avg_word_len': 0.0, 'word_count': 0
    }
    
    if not text: return features

    try:
        tagged_text = pos_tagger.tag(text)
        if not tagged_text: return features
            
        tokens = tagged_text.split()
        total_tokens = len(tokens)
        features['word_count'] = total_tokens
        
        if total_tokens == 0: return features

        clean_words = []
        for t in tokens:
            if '/' in t: clean_words.append(t.rsplit('/', 1)[0])
            else: clean_words.append(t)
                
        if clean_words:
            features['avg_word_len'] = sum(len(w) for w in clean_words) / len(clean_words)

        for token in tokens:
            if '/' not in token: continue
            tag = token.rsplit('/', 1)[1]
            
            if tag.startswith('S') or tag == 'NOUN' or tag == 'FOREIGN': features['NOUN_ratio'] += 1
            elif tag.startswith('V'): features['VERB_ratio'] += 1
            elif tag.startswith('PART') or tag in ['CONJ', 'PREP', 'PRON', 'H']: features['PART_ratio'] += 1
            elif tag.startswith('ADJ'): features['ADJ_ratio'] += 1
            elif tag.startswith('NUM') or tag == 'NSUFF': features['NUM_ratio'] += 1
            elif tag.startswith('PRON'): features['PRON_ratio'] += 1
            elif tag.startswith('DET'): features['DET_ratio'] += 1
            elif tag == 'PUNC': features['PUNC_ratio'] += 1

        for key in features:
            if key not in ['word_count', 'avg_word_len']:
                features[key] = features[key] / total_tokens

    except Exception:
        pass
        
    return features

print("Processing...")
tqdm.pandas()
features_df = df_final['text'].progress_apply(extract_features).apply(pd.Series)
final_dataset = pd.concat([features_df, df_final['label']], axis=1)
final_dataset = final_dataset[final_dataset['word_count'] > 0]

final_dataset.to_csv(OUTPUT_FILE, index=False)
print(f"Done. Feature file saved: '{OUTPUT_FILE}'")
