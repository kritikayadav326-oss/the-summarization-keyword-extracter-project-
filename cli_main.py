import argparse

from app import (
    fetch_article,
    summarize_text,
    extract_entities
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "The Summarization & Keyword Extractor"
        )
    )

    parser.add_argument(
        "--text",
        help="Article text"
    )

    parser.add_argument(
        "--url",
        help="Article URL"
    )

    parser.add_argument(
        "--summary-length",
        type=int,
        default=3,
        help="Number of summary sentences"
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of entities"
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Get article
    # ---------------------------------------------

    if args.text:

        article = args.text

    elif args.url:

        try:
            article = fetch_article(
                args.url
            )

        except Exception as error:

            print(
                f"Error fetching article: {error}"
            )

            return

    else:

        print(
            "Please provide either --text or --url."
        )

        return

    # ---------------------------------------------
    # Basic validation
    # ---------------------------------------------

    if len(article.split()) < 30:

        print(
            "Article is too short. "
            "Please provide at least 30 words."
        )

        return

    # ---------------------------------------------
    # Summarize
    # ---------------------------------------------

    summary = summarize_text(
        article,
        args.summary_length
    )

    # ---------------------------------------------
    # Extract entities
    # ---------------------------------------------

    entities = extract_entities(
        article,
        args.top_n
    )

    # ---------------------------------------------
    # Display results
    # ---------------------------------------------

    print("\n" + "=" * 60)

    print(
        "THE SUMMARIZATION & KEYWORD EXTRACTOR"
    )

    print("=" * 60)

    print("\nSUMMARY:")
    print(summary)

    print("\nTOP ENTITIES:")

    if entities:

        for (name, entity_type), count in entities:

            print(
                f"- {name} | "
                f"{entity_type} | "
                f"{count} occurrence(s)"
            )

    else:

        print(
            "No entities found."
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()