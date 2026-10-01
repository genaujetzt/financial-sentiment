import pandas as pd
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from preprocessing import preprocess_data


PATH = "data/financial_news.csv"

def prepare_test_data(df):
    """Создать фиксированный train/test split для анализа результатов"""
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    return train_idx, test_idx

def train_baseline(df):

    # Сначала train/test split, а то утекут данные ж
    train_idx, test_idx = prepare_test_data(df)

    X_train_text = df.loc[train_idx, "text_clean"]
    X_test_text = df.loc[test_idx, "text_clean"]

    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, vectorizer, y_test, y_pred

nlp = spacy.load('en_core_web_sm')

def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

    
def train_with_lem(df):

    train_idx, test_idx = prepare_test_data(df)

    X_train_text = df.loc[train_idx, "text_clean"]
    X_test_text = df.loc[test_idx, "text_clean"]

    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, vectorizer, y_test, y_pred


def train_trigram(df):

    train_idx, test_idx = prepare_test_data(df)

    X_train_text = df.loc[train_idx, "text_clean"]
    X_test_text = df.loc[test_idx, "text_clean"]

    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3)
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, vectorizer, y_test, y_pred


def train_forest(df):
    train_idx, test_idx = prepare_test_data(df)

    X_train_text = df.loc[train_idx, "text_clean"]
    X_test_text = df.loc[test_idx, "text_clean"]

    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, vectorizer, y_test, y_pred


def train_SVM(df):

    train_idx, test_idx = prepare_test_data(df)

    X_train_text = df.loc[train_idx, "text_clean"]
    X_test_text = df.loc[test_idx, "text_clean"]

    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]  

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LinearSVC(max_iter=2000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, vectorizer, y_test, y_pred

def train_hyperparam(df):

    train_idx, test_idx = prepare_test_data(df)

    X_train_text = df.loc[train_idx, "text_clean"]
    X_test_text = df.loc[test_idx, "text_clean"]

    y_train = df.loc[train_idx, "label"]
    y_test = df.loc[test_idx, "label"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    param_grid = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=3)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    return (
        grid.best_estimator_,
        vectorizer,
        y_test,
        y_pred,
        grid.best_params_
    )

if __name__ == "__main__":
    df = pd.read_csv(PATH)

    df = preprocess_data(df)

    model, vectorizer, y_test, y_pred = train_baseline(df)

    print("Baseline модель обучена.")
    print(f"Размер train: {len(y_test) * 4}")
    print(f"Размер test: {len(y_test)}")
    print(f"Количество TF-IDF признаков: {len(vectorizer.vocabulary_)}")