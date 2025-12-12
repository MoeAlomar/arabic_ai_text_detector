import pandas as pd
from farasa.pos import FarasaPOSTagger 
from tqdm import tqdm
import os
import sys
import re

# --- 1. الإعدادات ---
INPUT_FILE = "merged_dataset_clean2.csv"       
OUTPUT_FILE = "linguistic_features_augmented.csv" # ملف جديد للخصائص المطورة

COL_HUMAN = "human_collected_dataset"
COL_AI = "rephrased_text"

print("🚀 بدء استخراج الخصائص المطورة (Advanced Feature Engineering)...")

# --- 2. التحميل ---
if not os.path.exists(INPUT_FILE):
    print(f"❌ خطأ: الملف '{INPUT_FILE}' غير موجود.")
    exit()

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"📥 تم تحميل الملف: {len(df)} صف.")
except Exception as e:
    print(f"❌ خطأ في قراءة الملف: {e}")
    exit()

# تنظيف سريع
df = df.dropna(subset=[COL_HUMAN, COL_AI])
df = df[~df[COL_AI].astype(str).str.contains("SKIPPED|ERROR", na=False)]

# هيكلة
df_human = pd.DataFrame({'text': df[COL_HUMAN], 'label': 0})
df_ai = pd.DataFrame({'text': df[COL_AI], 'label': 1})
df_final = pd.concat([df_human, df_ai], ignore_index=True)
df_final['text'] = df_final['text'].astype(str)
df_final = df_final[df_final['text'].str.strip().str.len() > 10] 

print(f"📊 العينات: {len(df_final)}")

# --- 3. تهيئة Farasa ---
print("⏳ تشغيل Farasa...")
try:
    pos_tagger = FarasaPOSTagger(interactive=True)
except Exception as e:
    print(f"❌ فشل: {e}")
    exit()

# --- 4. دالة استخراج الخصائص (المطورة) ---
def extract_features(text):
    features = {
        'NOUN_ratio': 0.0, 'VERB_ratio': 0.0, 'PART_ratio': 0.0, 'ADJ_ratio': 0.0,
        'NUM_ratio': 0.0, 'PRON_ratio': 0.0, 'DET_ratio': 0.0, 'PUNC_ratio': 0.0,
        'avg_word_len': 0.0, 'word_count': 0,
        # 👇 خصائص جديدة لرفع الدقة
        'TTR_ratio': 0.0,       # التنوع المعجمي
        'avg_sentence_len': 0.0 # متوسط طول الجملة
    }
    
    if not text: return features

    try:
        # تقسيم الجمل (تقريبي بالنقطة)
        sentences = re.split(r'[.؟!]', text)
        sentences = [s for s in sentences if s.strip()]
        if sentences:
            features['avg_sentence_len'] = len(text.split()) / len(sentences)

        # التحليل الصرفي
        tagged_text = pos_tagger.tag(text)
        if not tagged_text: return features
            
        tokens = tagged_text.split()
        total_tokens = len(tokens)
        features['word_count'] = total_tokens
        
        if total_tokens == 0: return features

        # استخراج الكلمات الصافية (بدون تاجات)
        clean_words = []
        for t in tokens:
            if '/' in t: clean_words.append(t.rsplit('/', 1)[0])
            else: clean_words.append(t)
        
        # حساب التنوع المعجمي (عدد الكلمات الفريدة / العدد الكلي)
        if clean_words:
            features['TTR_ratio'] = len(set(clean_words)) / len(clean_words)
            features['avg_word_len'] = sum(len(w) for w in clean_words) / len(clean_words)

        # حساب النسب النحوية
        for token in tokens:
            if '/' not in token: continue
            tag = token.rsplit('/', 1)[1]
            
            if tag.startswith('S') or tag in ['NOUN', 'FOREIGN']: features['NOUN_ratio'] += 1
            elif tag.startswith('V'): features['VERB_ratio'] += 1
            elif tag.startswith('PART') or tag in ['CONJ', 'PREP', 'PRON', 'H']: features['PART_ratio'] += 1
            elif tag.startswith('ADJ'): features['ADJ_ratio'] += 1
            elif tag.startswith('NUM') or tag == 'NSUFF': features['NUM_ratio'] += 1
            elif tag.startswith('PRON'): features['PRON_ratio'] += 1
            elif tag.startswith('DET'): features['DET_ratio'] += 1
            elif tag == 'PUNC': features['PUNC_ratio'] += 1

        # التطبيع
        for key in features:
            if key not in ['word_count', 'avg_word_len', 'TTR_ratio', 'avg_sentence_len']:
                features[key] = features[key] / total_tokens

    except Exception:
        pass
        
    return features

# --- 5. التنفيذ ---
print("⚙️  جاري استخراج الخصائص المطورة...")
tqdm.pandas()
features_df = df_final['text'].progress_apply(extract_features).apply(pd.Series)
final_dataset = pd.concat([features_df, df_final['label']], axis=1)
final_dataset = final_dataset[final_dataset['word_count'] > 0]

final_dataset.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ تم الحفظ: '{OUTPUT_FILE}'")