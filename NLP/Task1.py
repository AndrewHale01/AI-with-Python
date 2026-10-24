import csv
import json
from pathlib import Path

import nltk
import stopwordsiso
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import wordpunct_tokenize
from simplemma import lemmatize
from textblob import TextBlob


INPUT_FILE = Path(__file__).with_name(
    "export_20261024T070500Z_1aadd924-d1d0-477f-b749-08e63f58c42a_Reviews.csv"
)
JSON_OUTPUT_FILE = Path(__file__).with_name("processed_reviews.json")
CSV_OUTPUT_FILE = Path(__file__).with_name("processed_reviews.csv")
SUMMARY_OUTPUT_FILE = Path(__file__).with_name("sentiment_summary.json")
CHART_OUTPUT_FILE = Path(__file__).with_name("sentiment_chart.png")

RUSSIAN_STEMMER = RussianStemmer()


def detect_language(text: str) -> str:
    """Distinguish Ukrainian and Russian by letters unique to each alphabet."""
    lowered = text.casefold()
    ukrainian_letters = sum(lowered.count(letter) for letter in "іїєґ")
    russian_letters = sum(lowered.count(letter) for letter in "ыэъё")
    if ukrainian_letters != russian_letters:
        return "uk" if ukrainian_letters > russian_letters else "ru"

    # Very short words without language-specific letters are ambiguous.
    return "uk"


def process_review(text: str, language: str) -> dict[str, list[str] | str]:
    tokens = [
        token.casefold()
        for token in wordpunct_tokenize(text)
        if token.isalpha()
    ]
    stop_words = (
        set(stopwords.words("russian"))
        if language == "ru"
        else stopwordsiso.stopwords("uk")
    )
    filtered_tokens = [token for token in tokens if token not in stop_words]

    if language == "ru":
        stemmed_tokens = [RUSSIAN_STEMMER.stem(token) for token in filtered_tokens]
        stemming_method = "NLTK Snowball Russian"
    else:
        stemmed_tokens = filtered_tokens.copy()
        stemming_method = "not available for Ukrainian in NLTK"

    lemmatized_tokens = [
        lemmatize(token, lang=language) for token in filtered_tokens
    ]
    return {
        "tokens": tokens,
        "tokens_without_stopwords": filtered_tokens,
        "stemmed": stemmed_tokens,
        "lemmatized": lemmatized_tokens,
        "stemming_method": stemming_method,
    }


def analyze_sentiment(text: str) -> dict[str, float | str]:
    sentiment = TextBlob(text).sentiment
    if sentiment.polarity > 0:
        label = "positive"
    elif sentiment.polarity < 0:
        label = "negative"
    else:
        label = "neutral"

    return {
        "sentiment": label,
        "sentiment_polarity": sentiment.polarity,
        "sentiment_subjectivity": sentiment.subjectivity,
    }


def main() -> None:
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"Не знайдено файл із відгуками: {INPUT_FILE}")
    if not nltk.download("stopwords", quiet=True):
        raise LookupError("Не вдалося завантажити корпус стоп-слів NLTK.")

    with INPUT_FILE.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "Review Text" not in reader.fieldnames:
            raise ValueError("У CSV має бути колонка 'Review Text'.")

        processed_rows = []
        for row in reader:
            review_text = (row.get("Review Text") or "").strip()
            language = detect_language(review_text)

            processed_rows.append(
                {
                    **row,
                    "language": language,
                    **process_review(review_text, language),
                    **analyze_sentiment(review_text),
                }
            )

    sentiment_counts = {
        label: sum(row["sentiment"] == label for row in processed_rows)
        for label in ("positive", "neutral", "negative")
    }

    with JSON_OUTPUT_FILE.open("w", encoding="utf-8") as json_file:
        json.dump(processed_rows, json_file, ensure_ascii=False, indent=2)

    output_fields = list(processed_rows[0]) if processed_rows else [
        "Task Link",
        "Reviewer",
        "Review Text",
        "language",
        "tokens",
        "tokens_without_stopwords",
        "stemmed",
        "lemmatized",
        "stemming_method",
        "sentiment",
        "sentiment_polarity",
        "sentiment_subjectivity",
    ]
    with CSV_OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=output_fields)
        writer.writeheader()
        for row in processed_rows:
            writer.writerow(
                {
                    **row,
                    "tokens": json.dumps(row["tokens"], ensure_ascii=False),
                    "tokens_without_stopwords": json.dumps(
                        row["tokens_without_stopwords"], ensure_ascii=False
                    ),
                    "stemmed": json.dumps(row["stemmed"], ensure_ascii=False),
                    "lemmatized": json.dumps(row["lemmatized"], ensure_ascii=False),
                }
            )

    with SUMMARY_OUTPUT_FILE.open("w", encoding="utf-8") as summary_file:
        json.dump(sentiment_counts, summary_file, ensure_ascii=False, indent=2)

    chart_labels = ["Позитивні", "Нейтральні", "Негативні"]
    chart_values = [
        sentiment_counts["positive"],
        sentiment_counts["neutral"],
        sentiment_counts["negative"],
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(
        chart_labels,
        chart_values,
        color=["#4CAF50", "#9E9E9E", "#F44336"],
    )
    axis.set_title("Тональність відгуків")
    axis.set_ylabel("Кількість відгуків")
    axis.set_ylim(0, max(chart_values, default=0) * 1.15 + 1)
    axis.bar_label(bars, padding=3)
    figure.tight_layout()
    figure.savefig(CHART_OUTPUT_FILE, dpi=150)
    plt.close(figure)

    print(f"Оброблено відгуків: {len(processed_rows)}")
    print(f"Позитивних: {sentiment_counts['positive']}")
    print(f"Нейтральних: {sentiment_counts['neutral']}")
    print(f"Негативних: {sentiment_counts['negative']}")
    print(f"JSON: {JSON_OUTPUT_FILE}")
    print(f"CSV: {CSV_OUTPUT_FILE}")
    print(f"Підсумок тональності: {SUMMARY_OUTPUT_FILE}")
    print(f"Діаграма: {CHART_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
