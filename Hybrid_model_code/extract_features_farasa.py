import pandas as pd
from farasa.pos import FarasaPOSTagger 
from tqdm import tqdm
import os
import sys

# --- 1. الإعدادات ---
INPUT_FILE = "merged_dataset_clean2.csv"       # ملفك المدمج ذو الـ 4 أعمدة
OUTPUT_FILE = "features_gemini_vs_human.csv"   # الملف الناتج للتدريب

# أسماء الأعمدة التي سنعتمد عليها فقط
COL_HUMAN = "human_collected_dataset"
COL_GEMINI = "gemini_rephrased_v2_5"  # نعتمد على هذا فقط كـ AI

print(f"🚀 بدء استخراج الخصائص: {COL_HUMAN} vs {COL_GEMINI}...")

# --- 2. تحميل البيانات ---
if not os.path.exists(INPUT_FILE):
    print(f"❌ خطأ: الملف '{INPUT_FILE}' غير موجود.")
    exit()

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"📥 تم تحميل الملف: {len(df)} صف.")
except Exception as e:
    print(f"❌ خطأ في قراءة الملف: {e}")
    exit()

# التأكد من وجود الأعمدة
if COL_HUMAN not in df.columns or COL_GEMINI not in df.columns:
    print(f"❌ الأعمدة المطلوبة غير موجودة.\nالموجود: {df.columns.tolist()}")
    exit()

# --- 3. تجهيز الهيكل (Text + Label) ---
print("🔄 تجاهل الأعمدة الأخرى (Qwen/Rewritten) والتركيز على Gemini...")

# نصوص بشرية (Label = 0)
df_human = pd.DataFrame({
    'text': df[COL_HUMAN],
    'label': 0
})

# نصوص Gemini (Label = 1)
df_gemini = pd.DataFrame({
    'text': df[COL_GEMINI],
    'label': 1
})

# دمجهم
df_final = pd.concat([df_human, df_gemini], ignore_index=True)

# تنظيف القيم الفارغة والنصوص القصيرة جداً
df_final.dropna(subset=['text'], inplace=True)
df_final['text'] = df_final['text'].astype(str)
df_final = df_final[df_final['text'].str.strip().str.len() > 5] 

print(f"📊 إجمالي العينات للمعالجة: {len(df_final)} عينة.")

# --- 4. تشغيل Farasa ---
print("⏳ جاري تشغيل Farasa POS Tagger...")
try:
    pos_tagger = FarasaPOSTagger(interactive=True)
    print("✅ تم التشغيل بنجاح.")
except Exception as e:
    print(f"❌ فشل تشغيل مكتبة فراسة: {e}")
    exit()

# --- 5. دالة استخراج الخصائص ---
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

# --- 6. التنفيذ ---
print("⚙️  جاري المعالجة...")
tqdm.pandas()
features_df = df_final['text'].progress_apply(extract_features).apply(pd.Series)
final_dataset = pd.concat([features_df, df_final['label']], axis=1)
final_dataset = final_dataset[final_dataset['word_count'] > 0]

# --- 7. الحفظ ---
final_dataset.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ تمت المهمة! ملف الخصائص جاهز: '{OUTPUT_FILE}'")