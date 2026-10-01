from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix

from train import (
    train_baseline,
    train_with_lem,
    train_trigram,
    train_forest,
    train_SVM,
    train_hyperparam,
    prepare_test_data
)
from preprocessing import preprocess_data
from save_model import save_model


DATA_PATH = "data/financial_news.csv"
RESULT_PATH = "reports/experiments.txt"
COMPARISON_PATH = "reports/results_table.txt"


def evaluate_model(
    name,
    model,
    vectorizer,
    y_test,
    y_pred,
    results
):

    report = classification_report(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    results[name] = f1    
    print(f"=== {name} ===")
    print(report)
    print(f"Macro F1: {f1:.4f}")

    with open(RESULT_PATH, "a") as f:
        f.write(f"=== {name} ===\n")
        f.write(report)
        f.write(f"\nMacro F1: {f1:.4f}\n\n")

    results[name] = f1

    return f1

def create_comparison_table(results, baseline_f1):

    baseline_f1 = results["BASELINE"]

    lines = [
        "| Метод | Macro F1 | Улучшение |",
        "|-------|----------|-----------|",
    ]

    for name, f1 in results.items():
        if name == "BASELINE":
            improvement = "-"
        else:
            improvement_value = (
                (f1 - baseline_f1) / baseline_f1 * 100
            )
            improvement = f"{improvement_value:+.2f}%"

        lines.append(
            f"| {name} | {f1:.4f} | {improvement} |"
        )

    return "\n".join(lines)


def update_best_model(
    f1,
    name,
    model,
    vectorizer,
    best_f1,
    best_model,
    best_vectorizer,
    best_name,
):

    if f1 > best_f1:
        return (
            f1,
            model,
            vectorizer,
            name,
        )

    return (
        best_f1,
        best_model,
        best_vectorizer,
        best_name,
    )


if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)

    train_idx, test_idx = prepare_test_data(df)

    test_texts = df.loc[test_idx, "text"]
    test_true_labels = df.loc[test_idx, "sentiment"]

    Path("reports").mkdir(exist_ok=True)
    Path(RESULT_PATH).write_text("")
    
    results = {}

    best_f1 = 0
    best_model = None
    best_vectorizer = None
    best_name = None

    experiments = [
        ("BASELINE", train_baseline),
        ("LEMMATIZATION", train_with_lem),
        ("TRIGRAM", train_trigram),
        ("RANDOM FOREST", train_forest),
        ("LINEAR SVC", train_SVM),
    ]
    
    for name, train_function in experiments:

        model, vectorizer, y_test, y_pred = train_function(df)

        f1 = evaluate_model(
            name,
            model,
            vectorizer,
            y_test,
            y_pred,
            results,
        )

        (
            best_f1,
            best_model,
            best_vectorizer,
            best_name,
        ) = update_best_model(
            f1,
            name,
            model,
            vectorizer,
            best_f1,
            best_model,
            best_vectorizer,
            best_name,
        )

    # GridSearch
    (
        model,
        vectorizer,
        y_test,
        y_pred,
        best_params,
    ) = train_hyperparam(df)

    f1 = evaluate_model(
        "LOGREG + GRIDSEARCH",
        model,
        vectorizer,
        y_test,
        y_pred,
        results,
    )

    with open(RESULT_PATH, "a") as f:
        f.write(f"Best parameters: {best_params}\n\n")

    (
        best_f1,
        best_model,
        best_vectorizer,
        best_name,
    ) = update_best_model(
        f1,
        "LOGREG + GRIDSEARCH",
        model,
        vectorizer,
        best_f1,
        best_model,
        best_vectorizer,
        best_name,
    )

    comparison_table = create_comparison_table(
    results,
    results["BASELINE"]
    )

    with open(COMPARISON_PATH, "w") as f:
        f.write(comparison_table)
        f.write("\n")

    print("\n=== Таблица результатов ===")
    print(comparison_table)

    save_model(best_model, best_vectorizer)

    print(f"\nЛучшая модель: {best_name}")
    print(f"Лучшая Macro F1: {best_f1:.4f}")

X_test_best = best_vectorizer.transform(
    df.loc[test_idx, "text_clean"]
)

y_test_best = df.loc[test_idx, "label"].values
y_pred_best = best_model.predict(X_test_best)


cm = confusion_matrix(y_test_best, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('reports/confusion_matrix.png')

X_test_best = best_vectorizer.transform(
    df.loc[test_idx, "text_clean"]
)

y_pred_best = best_model.predict(X_test_best)
y_test_best = df.loc[test_idx, "label"].values

errors = df.loc[test_idx].copy()
errors["predicted"] = y_pred_best

errors = errors[
    errors["sentiment"] != errors["predicted"]
]

print(f"\nКоличество ошибок: {len(errors)}")
print("\nПримеры ошибок:")

for _, row in errors.head(10).iterrows():
    print(f"\nТекст: {row['text']}")
    print(f"Истинный класс: {row['sentiment']}")
    print(f"Предсказанный класс: {row['predicted']}")

with open("reports/error_analysis.txt", "w") as f:
    f.write(f"Лучшая модель: {best_name}\n")
    f.write(f"Macro F1: {best_f1:.4f}\n")
    f.write(f"Количество ошибок: {len(errors)}\n\n")

    f.write("Примеры ошибок:\n\n")

    for _, row in errors.head(20).iterrows():
        f.write(f"Текст: {row['text']}\n")
        f.write(f"Истинный класс: {row['sentiment']}\n")
        f.write(f"Предсказанный класс: {row['predicted']}\n")
        f.write("-" * 80 + "\n")