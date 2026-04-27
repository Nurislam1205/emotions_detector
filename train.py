"""
Обучение модели Emotion Detection на датасете tweet_emotions.csv
Использует TF-IDF + LogisticRegression
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import re

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "tweet_emotions.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def clean_text(text):
    """Базовая очистка твита"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # убираем ссылки
    text = re.sub(r"@\w+", "", text)                      # убираем @mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # убираем # но оставляем слово
    text = re.sub(r"[^a-z\s']", " ", text)               # только буквы
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Файл не найден: {DATA_PATH}\n"
            "Скачай tweet_emotions.csv с Kaggle и положи в папку data/"
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Загружено строк: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")
    print(f"\nРаспределение эмоций:\n{df['sentiment'].value_counts()}\n")

    # Убираем строки без текста или метки
    df = df.dropna(subset=["content", "sentiment"])
    df["content"] = df["content"].apply(clean_text)
    df = df[df["content"].str.len() > 2]

    return df


def train():
    df = load_and_prepare_data()

    X = df["content"].values
    y = df["sentiment"].values

    # Кодируем метки
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Классы эмоций: {le.classes_.tolist()}")

    # Сначала split, потом oversample только train (честные метрики)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Оверсемплинг редких классов ТОЛЬКО в train
    MIN_SAMPLES = 600
    train_df = pd.DataFrame({"content": X_train, "label": y_train})
    parts = []
    for label_idx, group in train_df.groupby("label"):
        if len(group) < MIN_SAMPLES:
            oversampled = group.sample(MIN_SAMPLES, replace=True, random_state=42)
            parts.append(oversampled)
        else:
            parts.append(group)
    train_df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    X_train = train_df["content"].values
    y_train = train_df["label"].values
    print(f"Train после оверсемплинга: {len(X_train)} примеров\n")

    # TF-IDF векторизатор
    print("Обучаю TF-IDF векторизатор...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # LinearSVC обёрнутый в CalibratedClassifierCV для получения вероятностей
    print("Обучаю LinearSVC + калибровка...")
    svc = LinearSVC(C=0.5, max_iter=2000, class_weight="balanced")
    model = CalibratedClassifierCV(svc, cv=3)
    model.fit(X_train_tfidf, y_train)

    # Оценка
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nТочность на тестовой выборке: {acc:.4f} ({acc*100:.2f}%)")
    print("\nОтчёт по классам:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Сохраняем модель
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"\nМодель сохранена в {MODEL_DIR}/")


if __name__ == "__main__":
    train()
