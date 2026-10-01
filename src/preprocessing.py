import re
import pandas as pd

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_data(df):
    df = df.copy()


    df["text_clean"] = df["text"].apply(clean_text)


    df = df[df["text_clean"].str.len() > 0].copy()

 
    df["word_count_clean"] = df["text_clean"].str.split().str.len()
    df["char_count"] = df["text_clean"].str.len()
    df["dollar_count"] = df["text_clean"].str.count(r"\$")


    df["label"] = df["sentiment"].map({
        "negative": 0,
        "neutral": 1,
        "positive": 2
    })

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/financial_news.csv")
    df = preprocess_data(df)

    print(df.head())
    print(df[["sentiment", "label"]].head())