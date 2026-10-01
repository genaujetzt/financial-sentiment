import joblib
from pathlib import Path


MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def save_model(model, vectorizer):

    Path("models").mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"Vectorizer сохранён: {VECTORIZER_PATH}")