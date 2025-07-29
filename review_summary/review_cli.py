import string
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)
console = Console()


class ReviewSummary:
    def __init__(self, es):
        self.es = es

    def instructions(self):
        instructions = """
        # Yelp review summary tool

        Instructions
        1. To get a user's summary type, "user" <id>
        2. To exit, type "exit".
        """

        markdown = Markdown(instructions)
        console.print(markdown)

    def get_user_reviews_from_es(self, user_id, index_name="review_index"):
        query = {
            "query": {"term": {"user_id": user_id}},
            "aggs": {
                "review_count": {"value_count": {"field": "user_id"}},
                "unique_businesses": {"terms": {"field": "business_id", "size": 5000}},
            },
            "_source": ["business_id", "text"],
        }

        try:
            response = self.es.search(index=index_name, body=query)

            review_count = response["aggregations"]["review_count"]["value"]
            business_ids = [
                bucket["key"]
                for bucket in response["aggregations"]["unique_businesses"]["buckets"]
            ]

        except Exception as e:
            print(f"Error analyzing user reviews: {e}")
            raise

        # Extract hits (reviews)
        reviews = [hit["_source"] for hit in response["hits"]["hits"]]

        # Convert to DataFrame
        if reviews:
            return pd.DataFrame(reviews), review_count, business_ids
        else:
            return pd.DataFrame(), None, None

    def bounding_box(self, X, Y, r=10, R=6.4):
        C = 2 * np.pi * R

        dY = r * C / 360
        dX = dY * np.cos(np.radians(Y))

        Xmin, Ymin = X - dX, Y - dY
        Xmax, Ymax = X + dX, Y + dY
        return (Xmin, Ymin, Xmax, Ymax)

    def get_bounding_box(self, business_ids, index_name="business_data", top_n=10):

        response = self.es.search(
            index=index_name,
            query={
                "bool": {
                    "should": [
                        {"match": {"business_id": value}} for value in business_ids
                    ]
                }
            },
            size=top_n,
        )

        console.print(Markdown("**2. Bounding Boxes of businesses reviewed**\n"))
        bb_table = Table(show_header=True, header_style="bold magenta")
        bb_table.add_column("Rank", width=10)
        bb_table.add_column("ID", width=20)
        bb_table.add_column("Business Name", width=30)
        bb_table.add_column("Bounding Box", width=80)

        for i, hit in enumerate(response["hits"]["hits"]):
            bb_table.add_row(
                str(i + 1),
                hit["_id"],
                hit["_source"]["name"],
                str(
                    self.bounding_box(
                        hit["_source"]["longitude"], hit["_source"]["latitude"]
                    )
                ),
            )
        console.print(bb_table)
        print()

    # 3. Top 10 most frequent words (excluding stopwords)
    def get_top_words(self, user_reviews, top_n=10):
        all_reviews = " ".join(user_reviews["text"])
        tokens = word_tokenize(all_reviews.lower())
        # Filter out stopwords and punctuation
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        word_freq = Counter(tokens).most_common(top_n)

        console.print(Markdown("**3. Top 10 words used by user**\n"))
        tw_table = Table(show_header=True, header_style="bold magenta")
        tw_table.add_column("Rank", width=10)
        tw_table.add_column("Word", width=10)
        tw_table.add_column("Frequency", width=10)
        i = 0
        for word, freq in word_freq:
            i += 1
            tw_table.add_row(
                str(i),
                str(word),
                str(freq),
            )
        console.print(tw_table)
        print()

    # 4. Top 10 most frequent phrases (bi-grams)
    def get_top_phrases(self, user_reviews, top_n=10):
        all_reviews = " ".join(user_reviews["text"])
        tokens = word_tokenize(all_reviews.lower())
        # Filter out stopwords and punctuation
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        bi_grams = list(nltk.bigrams(tokens))
        phrase_freq = Counter(bi_grams).most_common(top_n)

        console.print(Markdown("**4. Top 10 phrases used by user**\n"))
        tp_table = Table(show_header=True, header_style="bold magenta")
        tp_table.add_column("Rank", width=10)
        tp_table.add_column("Phrase", width=20)
        tp_table.add_column("Frequency", width=10)
        i = 0
        for phrase, freq in phrase_freq:
            i += 1
            tp_table.add_row(
                str(i),
                " ".join(phrase),
                str(freq),
            )
        console.print(tp_table)
        print()

    # 5. Three most representative sentences
    def get_representative_sentences(self, user_reviews, top_n=3):
        all_reviews = " ".join(user_reviews["text"])
        sentences = sent_tokenize(all_reviews)

        # Simple approach: pick the three longest sentences (could be based on sentiment, frequency, etc.)
        sorted_sentences = sorted(sentences, key=len, reverse=True)[:top_n]

        console.print(Markdown("**5. Top  3 representative sentences**\n"))
        for i, sentence in enumerate(sorted_sentences, 1):
            print(f"{i}: {sentence}")
            print()

    def generate_user_review_summary(self, user_id):
        # Get user-specific reviews
        user_reviews, review_count, business_ids = self.get_user_reviews_from_es(
            user_id
        )

        if user_reviews.empty:
            print(f"No reviews found for user ID: {user_id}")
            return

        print(f"\n1. The user, {user_id}, has contributed {review_count} reviews.\n")

        self.get_bounding_box(business_ids)
        self.get_top_words(user_reviews)
        self.get_top_phrases(user_reviews)
        self.get_representative_sentences(user_reviews)


def test():
    index_name = "review_index"
    review_summary = ReviewSummary()

    while True:
        query = input("\nREVIEW SUMMARY QUERY: ").strip().split(" ")
        query[0] = query[0].lower()

        if query[0] == "exit":
            print("Exiting the search tool. Goodbye!")
            break

        if query[0] == "user":
            if len(query) < 2:
                print("No user id specified.")
                continue
            try:
                user_id = query[1]
                review_summary.generate_user_review_summary(index_name, user_id)
            except Exception as e:
                print(e)
        else:
            print(
                "Invalid search type. Please enter 'overall', 'user <id>', or 'exit'."
            )


if __name__ == "__main__":
    test()
