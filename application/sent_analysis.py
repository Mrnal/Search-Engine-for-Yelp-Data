import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from rich.console import Console
from rich.markdown import Markdown
from transformers import pipeline
from wordcloud import WordCloud

nltk.download("stopwords", quiet=True)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    batch_size=16,
)

nltk_stopwords = set(stopwords.words("english"))
console = Console()


class Application:
    def __init__(self, es):
        self.es = es

    def instructions(self):
        instructions = """
        # Yelp Business Performance Analysis tool

        Instructions
        1. To get the top positive and negative reviews associated with a business type the business_name.
        2. To exit, type "exit".
        """

        markdown = Markdown(instructions)
        console.print(markdown)

    def clean_and_tokenize(self, text):
        words = re.findall(r"\b\w+\b", text.lower())
        return [word for word in words if word not in nltk_stopwords]

    def generate_word_frequency(self, reviews):
        all_words = []
        for review in reviews:
            tokens = self.clean_and_tokenize(review)
            all_words.extend(tokens)

        word_count = Counter(all_words)
        return word_count

    def generate_visual_word_cloud(self, word_count):
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_count)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # Turn off axis labels
        plt.title("Word Cloud for Reviews")
        plt.show()

    def get_business_id(self, business_name):
        query = {"query": {"match": {"name": business_name}}}
        result = self.es.search(index="business_data", body=query)
        if result["hits"]["total"]["value"] > 0:
            business = result["hits"]["hits"][0]["_source"]
            return business["business_id"], business["name"]
        else:
            return None, None

    def get_reviews(self, business_id):
        query = {"query": {"term": {"business_id": business_id}}}
        result = self.es.search(index="review_index", body=query, size=1000)
        return [review["_source"]["text"] for review in result["hits"]["hits"]]

    def analyze_sentiments_batch(self, reviews):
        results = sentiment_analyzer(reviews)
        for i, review in enumerate(reviews):
            results[i]["text"] = review
        return results

    def classify_reviews_batch(self, reviews):
        sentiments = self.analyze_sentiments_batch(reviews)

        positive_reviews = [r for r in sentiments if r["label"] == "POSITIVE"]
        negative_reviews = [r for r in sentiments if r["label"] == "NEGATIVE"]

        top_positive_reviews = sorted(
            positive_reviews, key=lambda x: x["score"], reverse=True
        )
        top_negative_reviews = sorted(
            negative_reviews, key=lambda x: x["score"], reverse=True
        )

        return top_positive_reviews, top_negative_reviews

    def process_business_reviews(self, business_name):
        business_id, business_display_name = self.get_business_id(business_name)

        if business_id is None:
            print(f"No business found for name: {business_name}")
            return

        reviews = self.get_reviews(business_id)

        if not reviews:
            print(f"No reviews found for business: {business_display_name}")
            return

        word_count = self.generate_word_frequency(reviews)

        # Display visual word cloud
        print("\nGenerating Visual Word Cloud...")
        self.generate_visual_word_cloud(word_count)

        top_positive, top_negative = self.classify_reviews_batch(reviews)

        print(f"Business: {business_display_name}")

        print("\nTop 3 Positive Reviews:")
        for review in top_positive[:3]:
            print(f"Review: {review['text']}\nScore: {review['score']}\n")

        print("\nTop 3 Negative Reviews:")
        for review in top_negative[:3]:
            print(f"Review: {review['text']}\nScore: {review['score']}\n")

        if len(top_positive) > len(top_negative):
            print(f"{business_name} has more positive reviews than negative reviews")
            print()
        else:
            print(f"{business_name} has more negative reviews than positive reviews")
            print()


if __name__ == "__main__":
    business_name = input("Enter a business name: ")
    app = Application()
    app.process_business_reviews(business_name)
