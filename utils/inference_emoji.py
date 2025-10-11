# ============================================
# Inference Script (Emoji-Aware Sentiment Model)
# ============================================
# Usage:
#   python inference_emoji.py
# or import predict_sentiment() into another script.
#
# Requirements:
#   pip install Sastrawi emoji joblib scikit-learn pandas numpy

import re, joblib, glob, os, numpy as np
import emoji
from pathlib import Path
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Paths
OUTPUT_DIR = Path("/content/drive/MyDrive/Proyek/SentimentFinalEnhancedNew")

# --- Load the best model saved by the emoji-aware notebook ---
candidates = sorted(
    glob.glob(str(OUTPUT_DIR / "model_*_emoji.joblib")),
    key=os.path.getmtime,
    reverse=True
)
if not candidates:
    raise FileNotFoundError(f"No emoji model found in {OUTPUT_DIR}. Re-run training notebook first.")
MODEL_PATH = candidates[0]
print(f"Loaded model: {MODEL_PATH}")
pipe = joblib.load(MODEL_PATH)

# --- Recreate preprocessing (emoji + Sastrawi) ---
stemmer = StemmerFactory().create_stemmer()
stop_factory = StopWordRemoverFactory()
indo_stop = set(stop_factory.get_stop_words())
indo_stop |= {"yg","aja","kok","nih","sih","dong","nya","nyaa","krn","karna","karena","ttp","tetep"}

def basic_clean_with_emoji(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = emoji.demojize(text, delimiters=(" emoji_", ""))
    text = re.sub(r"[^a-z@#\\s_]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def simplify_emoji_tokens(text: str) -> str:
    text = re.sub(r"emoji_(?:face_with_tears_of_joy|grinning|smile|smiling_face|smiling_face_with_heart_eyes|heart_eyes|rolling_on_the_floor_laughing|sparkles|thumbs_up)", " emoji_positive ", text)
    text = re.sub(r"emoji_(?:angry|pouting|rage|cry|sob|disappointed|frowning|persevering_face|downcast_face_with_sweat|thumbs_down|weary)", " emoji_negative ", text)
    return text

def preprocess_id(text: str) -> str:
    t = basic_clean_with_emoji(text)
    t = simplify_emoji_tokens(t)
    tokens = [w for w in t.split() if w not in indo_stop]
    t = " ".join(tokens)
    t = stemmer.stem(t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t

def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]
    clean_texts = [preprocess_id(t) for t in texts]
    preds = pipe.predict(clean_texts)
    proba = pipe.predict_proba(clean_texts) if hasattr(pipe, "predict_proba") else None
    labels = pipe.classes_
    results = []
    for i, t in enumerate(texts):
        r = {
            "text": t,
            "clean": clean_texts[i],
            "pred": preds[i]
        }
        if proba is not None:
            top_idx = int(np.argmax(proba[i]))
            r["top_label"] = labels[top_idx]
            r["top_prob"] = float(proba[i][top_idx])
        results.append(r)
    return results

if __name__ == "__main__":
    samples = [
        "Mantap banget acaranya! 😂🔥 #keren",
        "Aduh… kecewa banget, nggak sesuai ekspektasi 😭",
        "Informasinya netral saja ya."
    ]
    results = predict_sentiment(samples)
    for r in results:
        print("\nText  :", r["text"])
        print("Clean :", r["clean"])
        print("Pred  :", r["pred"])
        if "top_label" in r:
            print("Top  :", r["top_label"], "| prob:", round(r["top_prob"], 3))
