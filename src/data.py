import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH = 'data/financial_news.csv'

def load_data():
    df = pd.read_csv(PATH)

    df = df.rename(columns={'sentence': 'text', 'label': 'sentiment'})

    df.to_csv("data/financial_news.csv", index=False)

    return df

def show_basic_info(df):
    print("Размер датасета:")
    print(df.shape)

    print("\nПервые 5 строк:")
    print(df.head())

    print("\nПропуски:")
    print(df.isnull().sum())

    print("\nРаспределение классов:")
    print(df["sentiment"].value_counts())

def save_class_distribution(df):
    sns.countplot(data=df, x="sentiment")
    plt.title("Распределение классов")
    plt.xlabel("Sentiment")
    plt.ylabel("Количество")

    plt.savefig(
        "reports/figures/sentiment_distribution.png",
        bbox_inches="tight"
    )

    plt.close()


def save_text_length_distribution(df):
    df["text_length"] = df["text"].str.len()

    df["text_length"].hist(bins=50)

    plt.xlabel("Длина текста")
    plt.ylabel("Количество")
    plt.title("Распределение длины текста")

    plt.savefig(
        "reports/figures/text_length_distribution.png",
        bbox_inches="tight"
    )

    plt.close()

def show_examples_by_sentiment(df):
    for sentiment in df["sentiment"].unique():
        print(f"\n=== Класс {sentiment} ===")

        examples = df[df["sentiment"] == sentiment]["text"].head(3)

        for example in examples:
            print(example)

    

if __name__ == "__main__":
    df = load_data()

    show_basic_info(df)
    show_examples_by_sentiment(df)

    save_class_distribution(df)
    save_text_length_distribution(df)

    print("\nГрафики сохранены:")
    print("reports/figures/sentiment_distribution.png")
    print("reports/figures/text_length_distribution.png")