import warnings

from dotenv import find_dotenv, load_dotenv
from nltk.corpus import wordnet as wn
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())

business_index = "business_data"
review_index = "review_index"

console = Console()


class SearchEngine:
    def __init__(self, es):
        self.es = es

    def instructions(self):
        instructions = """
        # Yelp search tool

        Search Types
        1. Business Name Search: Search for businesses by a keyword in the business name.
        2. Geo-spatial Search: Find businesses within a geographical bounding box based on latitude and longitude.

        Instructions
        1. For searching a business or a review you can just key in the phrase
        2. For searching businesses within a geo-spatial bounding box use a command of the following form "geo <top_lat> <top_lan> <bottom_lat> <bottom_lan>"
        """

        markdown = Markdown(instructions)
        console.print(markdown)

    def get_alternate_phrase(self, phrase):
        alternate_phrase = ""
        words = phrase.strip().split()
        for word in words:
            word_syn = wn.synsets(word)
            try:
                if word_syn is not None:
                    alternate_phrase += word + " "
                    continue
                alternate_phrase += word_syn[1].lemmas()[0].name() + " "
            except Exception as e:
                print("Error: ", e)
        return alternate_phrase.strip()

    def search_reviews(self, phrase, top_n=10):
        all_phrases = [phrase, self.get_alternate_phrase(phrase)]
        search_query = {
            "query": {
                "bool": {
                    "should": [{"match": {"text": _phrase}} for _phrase in all_phrases]
                }
            }
        }
        response = self.es.search(index=review_index, body=search_query, size=top_n)
        return response

    def search_business(self, phrase, top_n=10):
        all_phrases = [phrase, self.get_alternate_phrase(phrase)]
        search_query = {
            "query": {
                "bool": {
                    "should": [{"match": {"name": _phrase}} for _phrase in all_phrases]
                }
            }
        }
        response = self.es.search(index=business_index, body=search_query, size=top_n)
        return response

    def search(self, phrase):
        if len(phrase.strip().split()) == 1:
            response = self.search_business(phrase)
            console.print(Markdown("### Top business results\n"))

            # Use a table for better formatting
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", width=10)
            table.add_column("ID", width=20)
            table.add_column("Business Name", width=30)
            table.add_column("Address", width=40)
            table.add_column("Score", width=20)

            for i, hit in enumerate(response["hits"]["hits"]):
                table.add_row(
                    str(i + 1),
                    hit["_id"],
                    hit["_source"]["name"],
                    hit["_source"]["address"],
                    str(hit["_score"]),
                )
            console.print(table)
        else:
            reviews = self.search_reviews(phrase)
            business = self.search_business(phrase)

            reviews = sorted(
                reviews["hits"]["hits"], key=lambda rev: rev["_score"], reverse=True
            )
            business = sorted(
                business["hits"]["hits"], key=lambda bus: bus["_score"], reverse=True
            )

            console.print(Markdown("### Top business results\n"))

            # Business results table
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", width=10)
            table.add_column("ID", width=20)
            table.add_column("Business Name", width=30)
            table.add_column("Address", width=40)
            table.add_column("Score", width=20)
            for i, hit in enumerate(business):
                table.add_row(
                    str(i + 1),
                    hit["_id"],
                    hit["_source"]["name"],
                    hit["_source"]["address"],
                    str(hit["_score"]),
                )
            console.print(table)

            console.print(Markdown("\n### Top review results\n"))

            # Review results table
            review_table = Table(show_header=True, header_style="bold green")
            review_table.add_column("Rank", width=10)
            review_table.add_column("ID", width=20)
            review_table.add_column("Business Name", width=30)
            review_table.add_column("Review", width=70)
            review_table.add_column("Score", width=20)
            for i, hit in enumerate(reviews):
                business_name = self.es.get(
                    index=business_index, id=hit["_source"]["business_id"]
                )["_source"]["name"]
                review_table.add_row(
                    str(i + 1),
                    hit["_id"],
                    business_name,
                    hit["_source"]["text"],
                    str(hit["_score"]),
                )
            console.print(review_table)

    def search_business_by_location(
        self, top_left, bottom_right, top_n=10, index_name="business_data"
    ):
        search_query = {
            "query": {
                "geo_bounding_box": {
                    "location": {"top_left": top_left, "bottom_right": bottom_right}
                }
            }
        }
        response = self.es.search(index=index_name, body=search_query, size=top_n)
        print("\nSearch Results for Location:")
        for hit in response["hits"]["hits"]:
            print(
                f"Name: {hit['_source']['name']}, Location: {hit['_source']['location']}"
            )


def test():
    index_name = "business_data"
    search = SearchEngine()

    print("Yelp search tool for businesses and reviews")
    print("Instructions")
    print("Type 'exit' to quit at any time.\n")

    while True:
        query = input("query: ").strip().lower()
        query = query.split(" ")
        if query[0] == "exit":
            print("Exiting the search tool. Goodbye!")
            break
        elif query[0] != "geo":
            search.search(" ".join(query).strip())
        elif query[0] == "geo":
            if len(query) < 5:
                print("geo query passed with few params :(")
                continue

            for q in query[1:]:
                try:
                    _ = float(q)
                except:
                    print("lat lon values are not real valued numbers :(")
                    continue

            query[1:] = [float(q) for q in query[1:]]
            try:
                top_left = {"lat": query[1], "lon": query[2]}
                bottom_right = {"lat": query[3], "lon": query[4]}
                search.search_business_by_location(index_name, top_left, bottom_right)
            except Exception as e:
                print(e)
        else:
            print("Invalid search type. Please enter 'name', 'geo', or 'exit'.")

        print("\n")


if __name__ == "__main__":
    test()
