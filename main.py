import os
import warnings

from dotenv import find_dotenv, load_dotenv
from elasticsearch import Elasticsearch
from rich import print
from rich.console import Console
from rich.markdown import Markdown

from application import sent_analysis
from review_summary import review_cli
from search_engine import cli

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())

console = Console()
instructions = """
# Yelp search tool for businesses and reviews

## Instructions
1. For the search engine, type "search"
2. For the review summary type "review"
3. For application type "app"
4. To exit, type "exit".
"""

markdown = Markdown(instructions)
console.print(markdown)


def setup():
    try:
        es = Elasticsearch(
            api_key=os.environ.get("API_KEY"), cloud_id=os.environ.get("CLOUD_ID")
        )
        return es
    except Exception as e:
        print("Error:", e)
        raise


def review(es):
    review_summary = review_cli.ReviewSummary(es)
    review_summary.instructions()

    while True:
        query = input("REVIEW SUMMARY: ").strip().split(" ")
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
                review_summary.generate_user_review_summary(user_id)
            except Exception as e:
                print(e)
        else:
            print(
                "Invalid search type. Please enter 'overall', 'user <id>', or 'exit'."
            )


def business(es):
    search_engine = cli.SearchEngine(es)
    search_engine.instructions()
    while True:
        query = input("SEARCH: ").strip().lower()
        query = query.split(" ")
        if query[0] == "exit":
            print("Exiting the search tool. Goodbye!")
            break
        elif query[0] != "geo":
            search_engine.search(" ".join(query).strip())
        elif query[0] == "geo":
            if len(query) < 5:
                print("geo query passed with few params :(")
                continue

            for q in query[1:]:
                try:
                    _ = float(q)
                except Exception:
                    print("lat lon values are not real valued numbers :(")
                    continue

            query[1:] = [float(q) for q in query[1:]]
            try:
                top_left = {"lat": query[1], "lon": query[2]}
                bottom_right = {"lat": query[3], "lon": query[4]}
                search_engine.search_business_by_location(top_left, bottom_right)
            except Exception as e:
                print(e)
        else:
            print("Invalid search type. Please enter 'name', 'geo', or 'exit'.")


def app(es):
    sent_app = sent_analysis.Application(es)
    sent_app.instructions()

    while True:
        query = input("BUSINESS PERFORMANCE ANALYSIS: ").strip().lower()

        if query == "exit":
            print("Exiting the search tool. Goodbye!")
            break

        sent_app.process_business_reviews(query)


def main():
    es = setup()

    while True:
        query = input("QUERY: ").strip().lower().split()

        if len(query) > 1:
            print(
                "Invalid search type. Please enter either one of 'search', 'review', 'app' or 'exit'."
            )

        if query[0] == "exit":
            print("Exiting the search tool. Goodbye!")
            break

        if query[0] == "review":
            review(es)
            print()
        elif query[0] == "search":
            business(es)
            print()
        elif query[0] == "app":
            app(es)
            print()
        else:
            print(
                "Invalid search type. Please enter either one of 'search', 'review' or 'exit'."
            )


if __name__ == "__main__":
    main()
