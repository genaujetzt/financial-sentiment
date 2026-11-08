import joblib

from pathlib import Path
from preprocessing import clean_text


MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def predict_sentiment(texts, model, vectorizer, clean_func=None):
    if isinstance(texts, str):
        texts = [texts]

    if clean_func:
        texts = [clean_func(t) for t in texts]

    X = vectorizer.transform(texts)
    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        return predictions, probs

    return predictions


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    texts = [
        "The company reported strong growth in revenue.",
        "The company announced a significant loss this quarter.",
        "The company expects stable results next year.",
        "Sales increased by 15 percent compared with last year.",
        "The company plans to reduce its workforce."
    ]
    result = predict_sentiment(
    texts,
    model,
    vectorizer,
    clean_text)
    
    Path("reports").mkdir(parents=True, exist_ok=True)

    with open("reports/predictions.txt", "w") as f:

        if isinstance(result, tuple):
            predictions, probabilities = result

            for text, prediction, probability in zip(
                texts,
                predictions,
                probabilities
            ):
                f.write(f"Text: {text}\n")
                f.write(f"Prediction: {prediction}\n")
                f.write(f"Probabilities: {probability}\n")
                f.write("-" * 80 + "\n")

        else:
            for text, prediction in zip(texts, result):
                f.write(f"Text: {text}\n")
                f.write(f"Prediction: {prediction}\n")
                f.write("-" * 80 + "\n")

    print("Результаты сохранены в reports/predictions.txt")