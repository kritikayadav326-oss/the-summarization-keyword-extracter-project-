import re
from collections import Counter

import requests
import spacy
import streamlit as st
from bs4 import BeautifulSoup


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Summarization & Keyword Extractor",
    page_icon="📰",
    layout="wide"
)


# ---------------------------------------------------------
# Load spaCy model
# ---------------------------------------------------------

@st.cache_resource
def load_nlp_model():
    """
    Load the spaCy English model.
    The model is loaded only once.
    """
    return spacy.load("en_core_web_sm")


# ---------------------------------------------------------
# URL validation
# ---------------------------------------------------------

def is_valid_url(url):
    """
    Basic URL validation.
    """
    return (
        url.startswith("http://")
        or url.startswith("https://")
    )


# ---------------------------------------------------------
# Fetch article from URL
# ---------------------------------------------------------

def fetch_article(url):
    """
    Download a webpage and extract paragraph text.
    """

    if not is_valid_url(url):
        raise ValueError(
            "Please enter a valid URL starting with http:// or https://"
        )

    try:
        response = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise ValueError(
            f"Could not fetch the webpage: {error}"
        )

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Remove unnecessary webpage elements
    for tag in soup(
        ["script", "style", "nav", "footer", "header"]
    ):
        tag.decompose()

    # Get paragraph text
    paragraphs = soup.find_all("p")

    article_text = " ".join(
        paragraph.get_text(" ", strip=True)
        for paragraph in paragraphs
    )

    # Clean extra spaces
    article_text = re.sub(
        r"\s+",
        " ",
        article_text
    ).strip()

    if len(article_text.split()) < 30:
        raise ValueError(
            "Could not find enough article text on this webpage."
        )

    return article_text


# ---------------------------------------------------------
# Split text into sentences
# ---------------------------------------------------------

def split_sentences(text):
    """
    Simple sentence splitting using punctuation.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ---------------------------------------------------------
# Simple extractive summarization
# ---------------------------------------------------------

def summarize_text(text, number_of_sentences):
    """
    Create a simple extractive summary.

    The algorithm:
    1. Split article into sentences.
    2. Count important words.
    3. Score each sentence.
    4. Select the highest-scoring sentences.
    """

    sentences = split_sentences(text)

    if not sentences:
        return "No sentences found."

    # If article already has fewer sentences than requested
    if len(sentences) <= number_of_sentences:
        return " ".join(sentences)

    # Common English words that should not influence scoring
    stop_words = {
        "the", "a", "an", "and", "or", "but",
        "is", "are", "was", "were", "to",
        "of", "in", "on", "for", "with",
        "that", "this", "as", "by", "at",
        "from", "it", "its", "be", "has",
        "have", "had", "will", "would",
        "can", "could", "about", "into",
        "their", "they", "them", "he", "she",
        "his", "her", "we", "our", "you",
        "your", "I"
    }

    # Extract words from article
    words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    # Remove common words
    important_words = [
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    ]

    # Count word frequency
    word_frequency = Counter(
        important_words
    )

    # Score every sentence
    sentence_scores = []

    for index, sentence in enumerate(sentences):

        sentence_words = re.findall(
            r"\b[a-zA-Z]+\b",
            sentence.lower()
        )

        score = sum(
            word_frequency[word]
            for word in sentence_words
            if word in word_frequency
        )

        # Small bonus for early sentences.
        # News articles often put important information near the beginning.
        if index < 2:
            score *= 1.2

        sentence_scores.append(
            (score, index, sentence)
        )

    # Select highest-scoring sentences
    sentence_scores.sort(
        reverse=True
    )

    selected = sentence_scores[
        :number_of_sentences
    ]

    # Restore original article order
    selected.sort(
        key=lambda item: item[1]
    )

    summary = " ".join(
        sentence
        for _, _, sentence in selected
    )

    return summary


# ---------------------------------------------------------
# Named Entity Recognition
# ---------------------------------------------------------

def extract_entities(text, top_n):
    """
    Extract People, Organizations and Locations
    using spaCy.
    """

    nlp = load_nlp_model()

    doc = nlp(text)

    entities = []

    for entity in doc.ents:

        if entity.label_ in [
            "PERSON",
            "ORG",
            "GPE",
            "LOC"
        ]:

            if entity.label_ == "PERSON":
                entity_type = "Person"

            elif entity.label_ == "ORG":
                entity_type = "Organization"

            else:
                entity_type = "Location"

            entities.append(
                (
                    entity.text,
                    entity_type
                )
            )

    # Count duplicate entities
    entity_counts = Counter(
        entities
    )

    # Sort by frequency
    top_entities = entity_counts.most_common(
        top_n
    )

    return top_entities


# ---------------------------------------------------------
# Main Streamlit application
# ---------------------------------------------------------

def main():

    st.title(
        "📰 The Summarization & Keyword Extractor"
    )

    st.write(
        "Enter a news article and get a short summary "
        "along with important people, organizations and locations."
    )

    # -----------------------------------------------------
    # Sidebar controls
    # -----------------------------------------------------

    st.sidebar.header("⚙️ Settings")

    summary_length = st.sidebar.slider(
        "Summary Length",
        min_value=1,
        max_value=10,
        value=3
    )

    top_entities = st.sidebar.slider(
        "Top N Entities",
        min_value=1,
        max_value=10,
        value=5
    )

    # -----------------------------------------------------
    # Input tabs
    # -----------------------------------------------------

    paste_tab, url_tab = st.tabs(
        ["📝 Paste Text", "🌐 Fetch via URL"]
    )

    article_text = ""

    # -----------------------------------------------------
    # Paste Text
    # -----------------------------------------------------

    with paste_tab:

        text_input = st.text_area(
            "Paste your article here:",
            height=300,
            placeholder=(
                "Paste a news article into this box..."
            )
        )

        if text_input.strip():
            article_text = text_input

    # -----------------------------------------------------
    # Fetch URL
    # -----------------------------------------------------

    with url_tab:

        url_input = st.text_input(
            "Enter news article URL:",
            placeholder="https://example.com/article"
        )

        if st.button("Fetch Article"):

            if not url_input.strip():

                st.warning(
                    "Please enter a URL."
                )

            else:

                try:

                    with st.spinner(
                        "Fetching article..."
                    ):

                        article_text = fetch_article(
                            url_input
                        )

                        st.session_state[
                            "article_text"
                        ] = article_text

                    st.success(
                        "Article fetched successfully!"
                    )

                except ValueError as error:

                    st.error(str(error))

        # Get previously fetched article
        if "article_text" in st.session_state:

            article_text = st.session_state[
                "article_text"
            ]

            st.text_area(
                "Extracted Article:",
                value=article_text,
                height=250,
                disabled=True
            )

    # -----------------------------------------------------
    # Analyze button
    # -----------------------------------------------------

    st.divider()

    if st.button(
        "🚀 Summarize & Extract Keywords",
        type="primary",
        use_container_width=True
    ):

        if not article_text.strip():

            st.warning(
                "Please paste an article or fetch one using a URL."
            )

            return

        if len(article_text.split()) < 30:

            st.warning(
                "Please provide an article with at least 30 words."
            )

            return

        try:

            with st.spinner(
                "Analyzing article..."
            ):

                summary = summarize_text(
                    article_text,
                    summary_length
                )

                entities = extract_entities(
                    article_text,
                    top_entities
                )

            # -------------------------------------------------
            # Display summary
            # -------------------------------------------------

            st.subheader(
                "📌 Summary"
            )

            st.info(summary)

            # -------------------------------------------------
            # Display entities
            # -------------------------------------------------

            st.subheader(
                "🔎 Important Entities"
            )

            if entities:

                # Create table data
                table_data = []

                for (name, entity_type), count in entities:

                    table_data.append(
                        {
                            "Entity": name,
                            "Type": entity_type,
                            "Frequency": count
                        }
                    )

                st.table(
                    table_data
                )

            else:

                st.info(
                    "No People, Organizations or Locations were found."
                )

            # -------------------------------------------------
            # Basic statistics
            # -------------------------------------------------

            st.subheader(
                "📊 Article Statistics"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Words",
                    len(article_text.split())
                )

            with col2:
                st.metric(
                    "Sentences",
                    len(split_sentences(article_text))
                )

            with col3:
                st.metric(
                    "Entities",
                    len(entities)
                )

        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )


# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()