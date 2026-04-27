"""
Flask приложение — Emotion Detector
Режим 1: Локальная модель (TF-IDF + LogisticRegression)
Режим 2: Gemini API (бесплатный)
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Иконки и цвета для каждой эмоции ──────────────────────────────────────────
EMOTION_META = {
    "anger":       {"icon": "😠", "color": "#e74c3c", "label": "Злость"},
    "boredom":     {"icon": "😑", "color": "#95a5a6", "label": "Скука"},
    "enthusiasm":  {"icon": "🤩", "color": "#f39c12", "label": "Энтузиазм"},
    "empty":       {"icon": "😶", "color": "#bdc3c7", "label": "Пустота"},
    "fun":         {"icon": "😄", "color": "#2ecc71", "label": "Веселье"},
    "happiness":   {"icon": "😊", "color": "#27ae60", "label": "Счастье"},
    "hate":        {"icon": "🤬", "color": "#c0392b", "label": "Ненависть"},
    "love":        {"icon": "❤️",  "color": "#e91e63", "label": "Любовь"},
    "neutral":     {"icon": "😐", "color": "#7f8c8d", "label": "Нейтрально"},
    "relief":      {"icon": "😌", "color": "#1abc9c", "label": "Облегчение"},
    "sadness":     {"icon": "😢", "color": "#3498db", "label": "Грусть"},
    "surprise":    {"icon": "😲", "color": "#9b59b6", "label": "Удивление"},
    "worry":       {"icon": "😟", "color": "#e67e22", "label": "Тревога"},
}


def get_emotion_meta(emotion_key):
    key = emotion_key.lower()
    return EMOTION_META.get(key, {"icon": "🤔", "color": "#555", "label": emotion_key.capitalize()})


# ── Загрузка локальной модели ──────────────────────────────────────────────────
_model = None
_vectorizer = None
_label_encoder = None
_model_loaded = False


def load_local_model():
    global _model, _vectorizer, _label_encoder, _model_loaded
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if all(os.path.exists(p) for p in [model_path, vec_path, le_path]):
        _model = joblib.load(model_path)
        _vectorizer = joblib.load(vec_path)
        _label_encoder = joblib.load(le_path)
        _model_loaded = True
        print("Локальная модель загружена.")
    else:
        print("Файлы модели не найдены. Запусти python train.py сначала.")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_local(text):
    if not _model_loaded:
        return None, None, "Модель не загружена. Запусти python train.py"

    cleaned = clean_text(text)
    vec = _vectorizer.transform([cleaned])
    pred_idx = _model.predict(vec)[0]
    proba = _model.predict_proba(vec)[0]

    emotion = _label_encoder.classes_[pred_idx]
    confidence = float(proba[pred_idx])

    # Топ-3 предсказания
    top3_idx = np.argsort(proba)[::-1][:3]
    top3 = [
        {
            "emotion": _label_encoder.classes_[i],
            "prob": round(float(proba[i]) * 100, 1),
            **get_emotion_meta(_label_encoder.classes_[i]),
        }
        for i in top3_idx
    ]

    return emotion, confidence, None, top3


# ── Gemini API ─────────────────────────────────────────────────────────────────
def predict_gemini(text, api_key):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "Установи пакет: pip install google-genai"

    if not api_key or not api_key.strip():
        return None, "API ключ не настроен в .env файле"

    client = genai.Client(api_key=api_key.strip())

    emotion_list = ", ".join(EMOTION_META.keys())
    prompt = f"""Analyze the emotion in the following text and respond ONLY with valid JSON.

Text: "{text}"

Choose the single best emotion from this list: {emotion_list}

Respond with this exact JSON format (nothing else):
{{
  "emotion": "<emotion from list>",
  "confidence": <number 0-100>,
  "explanation": "<one sentence explanation in Russian>"
}}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
        )
        raw = response.text.strip()

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return None, f"Не удалось разобрать ответ Gemini: {raw}"

        result = json.loads(json_match.group())
        emotion = result.get("emotion", "neutral").lower()
        confidence = float(result.get("confidence", 0)) / 100
        explanation = result.get("explanation", "")

        meta = get_emotion_meta(emotion)
        return {
            "emotion": emotion,
            "confidence": confidence,
            "explanation": explanation,
            **meta,
            "top3": [{"emotion": emotion, "prob": round(confidence * 100, 1), **meta}],
        }, None

    except Exception as e:
        return None, f"Ошибка Gemini API: {str(e)}"


# ── Маршруты Flask ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_loaded=_model_loaded)


@app.route("/predict/local", methods=["POST"])
def predict_local_route():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Введи текст"}), 400

    emotion, confidence, error, top3 = predict_local(text)

    if error:
        return jsonify({"error": error}), 500

    meta = get_emotion_meta(emotion)
    return jsonify({
        "emotion": emotion,
        "confidence": round(confidence * 100, 1),
        "top3": top3,
        **meta,
    })


@app.route("/predict/gemini", methods=["POST"])
def predict_gemini_route():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Введи текст"}), 400

    # Ключ берётся с сервера из .env — не из запроса
    result, error = predict_gemini(text, GEMINI_API_KEY)

    if error:
        return jsonify({"error": error}), 400

    return jsonify({
        "emotion": result["emotion"],
        "confidence": round(result["confidence"] * 100, 1),
        "label": result["label"],
        "icon": result["icon"],
        "color": result["color"],
        "explanation": result.get("explanation", ""),
        "top3": result.get("top3", []),
    })


@app.route("/model/status")
def model_status():
    return jsonify({"loaded": _model_loaded})


@app.route("/stats")
def stats():
    data_path = os.path.join(os.path.dirname(__file__), "data", "tweet_emotions.csv")
    if not os.path.exists(data_path):
        return jsonify({"error": "Датасет не найден"}), 404

    df = pd.read_csv(data_path).dropna(subset=["content", "sentiment"])

    colors = [EMOTION_META.get(e, {}).get("color", "#888") for e in df["sentiment"].value_counts().index]
    labels_ru = [EMOTION_META.get(e, {}).get("label", e) for e in df["sentiment"].value_counts().index]

    # 1. Распределение эмоций
    dist = df["sentiment"].value_counts()

    # 2. Средняя длина текста по эмоциям
    df["text_len"] = df["content"].astype(str).apply(lambda x: len(x.split()))
    avg_len = df.groupby("sentiment")["text_len"].mean().round(1)

    # 3. Топ-10 слов на весь датасет (стоп-слова убраны)
    STOPWORDS = {"i", "the", "a", "to", "and", "is", "it", "in", "of", "my",
                 "me", "so", "im", "for", "on", "you", "at", "be", "are",
                 "was", "that", "have", "just", "with", "but", "not", "this",
                 "its", "get", "got", "do", "dont", "am", "up", "he", "she",
                 "they", "we", "or", "an", "as", "all", "from", "if", "about"}
    words = []
    for txt in df["content"].astype(str):
        words.extend([w.lower() for w in re.findall(r"[a-zA-Z']+", txt) if w.lower() not in STOPWORDS and len(w) > 2])
    top_words = Counter(words).most_common(10)

    # 4. Длины текстов (гистограмма бакетами)
    buckets = [0, 5, 10, 20, 30, 50, 100, 9999]
    labels_len = ["1-5", "6-10", "11-20", "21-30", "31-50", "51-100", "100+"]
    counts_len = []
    for i in range(len(buckets) - 1):
        c = ((df["text_len"] > buckets[i]) & (df["text_len"] <= buckets[i+1])).sum()
        counts_len.append(int(c))

    # 5. Positive / Negative / Neutral grouping
    POS = {"happiness", "love", "fun", "enthusiasm", "relief", "surprise"}
    NEG = {"sadness", "anger", "hate", "worry", "boredom"}
    NEU = {"neutral", "empty"}
    def group(e):
        if e in POS: return "Позитивные"
        if e in NEG: return "Негативные"
        return "Нейтральные"
    df["group"] = df["sentiment"].apply(group)
    grp = df["group"].value_counts()

    # 6. Top-5 words per emotion
    top_per_emotion = {}
    for emotion, edf in df.groupby("sentiment"):
        w = []
        for txt in edf["content"].astype(str):
            w += [x.lower() for x in re.findall(r"[a-zA-Z']+", txt)
                  if x.lower() not in STOPWORDS and len(x) > 2]
        top5 = Counter(w).most_common(7)
        meta = EMOTION_META.get(emotion, {})
        top_per_emotion[emotion] = {
            "words": [x for x, _ in top5],
            "counts": [c for _, c in top5],
            "color": meta.get("color", "#888"),
            "label": meta.get("label", emotion),
        }

    # 7. Length percentiles per emotion (p25, median, p75)
    perc = df.groupby("sentiment")["text_len"].quantile([0.25, 0.5, 0.75]).unstack()
    perc_emotions = list(perc.index)
    return jsonify({
        "total": int(len(df)),
        "distribution": {
            "labels": list(dist.index),
            "labels_ru": labels_ru,
            "counts": [int(x) for x in dist.values],
            "colors": colors,
        },
        "avg_length": {
            "labels": list(avg_len.index),
            "labels_ru": [EMOTION_META.get(e, {}).get("label", e) for e in avg_len.index],
            "values": [float(x) for x in avg_len.values],
            "colors": [EMOTION_META.get(e, {}).get("color", "#888") for e in avg_len.index],
        },
        "top_words": {
            "words": [w for w, _ in top_words],
            "counts": [c for _, c in top_words],
        },
        "text_lengths": {
            "labels": labels_len,
            "counts": counts_len,
        },
        "sentiment_groups": {
            "labels": list(grp.index),
            "counts": [int(x) for x in grp.values],
            "colors": ["#27ae60", "#e74c3c", "#7f8c8d"],
        },
        "top_per_emotion": top_per_emotion,
        "length_percentiles": {
            "emotions": perc_emotions,
            "labels_ru": [EMOTION_META.get(e, {}).get("label", e) for e in perc_emotions],
            "p25":    [float(perc.loc[e, 0.25]) for e in perc_emotions],
            "median": [float(perc.loc[e, 0.50]) for e in perc_emotions],
            "p75":    [float(perc.loc[e, 0.75]) for e in perc_emotions],
            "colors": [EMOTION_META.get(e, {}).get("color", "#888") for e in perc_emotions],
        },
    })


if __name__ == "__main__":
    load_local_model()
    app.run(debug=False, host="0.0.0.0", port=5000)
