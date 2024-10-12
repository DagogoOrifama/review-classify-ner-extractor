import streamlit as st
import spacy
from spacy.matcher import Matcher
from spacy import displacy
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import tensorflow as tf

# Load custom model and vectorizer from Hugging Face Hub
model_path = hf_hub_download(repo_id="dagorislab/ds-opinion-classifier", filename="model_3.h5")
model_3 = tf.keras.models.load_model(model_path)

# Load the TF-IDF vectorizer
vectorizer_path = hf_hub_download(repo_id="dagorislab/ds-opinion-classifier", filename="tfidf_vectorizer.pkl")
tfidf_vectorizer = joblib.load(vectorizer_path)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to set up the custom matcher
def setup_custom_matcher(nlp):
    matcher = Matcher(nlp.vocab)
    product_patterns = [
        [{"LOWER": "fire"}, {"LOWER": "hd"}, {"IS_DIGIT": True}],
        [{"LOWER": "fire"}, {"LOWER": "tablet"}],
        [{"LOWER": "tablet"}],
        [{"LOWER": "kindle"}, {"LOWER": "fire"}],
        [{"LOWER": "kindle"}]
    ]
    part_patterns = [
        [{"LOWER": "screen"}],
        [{"LOWER": "display"}],
        [{"LOWER": "camera"}],
        [{"LOWER": "battery"}],
        [{"LOWER": "charger"}],
        [{"LOWER": "case"}],
        [{"LOWER": "google"}, {"LOWER": "play"}, {"LOWER": "store"}],
        [{"LOWER": "amazon"}, {"LOWER": "app"}, {"LOWER": "store"}],
        [{"LOWER": "pin"}, {"LOWER": "protected"}, {"LOWER": "parent"}, {"LOWER": "lock"}],
        [{"LOWER": "navigation"}],
        [{"LOWER": "settings"}],
        [{"LOWER": "night"}, {"LOWER": "mode"}],
        [{"LOWER": "wifi"}],
        [{"LOWER": "connectivity"}]
    ]
    matcher.add("Product", product_patterns)
    matcher.add("Part", part_patterns)
    return matcher

# Initialize the matcher
matcher = setup_custom_matcher(nlp)

# Function to retrieve and visualize entities in a review
def retrieve_and_visualize_entities(review, nlp, matcher):
    doc = nlp(review)
    matches = matcher(doc)
    entities = set()
    spans = []
    for match_id, start, end in matches:
        entity_label = nlp.vocab.strings[match_id]
        entity_text = doc[start:end].text
        entities.add((entity_text, entity_label))
        spans.append({"start": doc[start:end].start_char, "end": doc[start:end].end_char, "label": entity_label})
    displacy.render({"text": doc.text, "ents": spans, "title": None}, style="ent", manual=True, jupyter=False)
    return list(entities)

# Function to classify a review
def classify_review(review, model, tfidf_vectorizer):
    review_tfidf = tfidf_vectorizer.transform([review])
    prediction = model.predict(review_tfidf.toarray())
    predicted_class = prediction.argmax(axis=1)[0]
    class_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return class_mapping[predicted_class]

# Unified function to retrieve entities, visualize, and classify a review
def analyze_review(review, nlp, matcher, model, tfidf_vectorizer):
    entities = retrieve_and_visualize_entities(review, nlp, matcher)
    sentiment = classify_review(review, model, tfidf_vectorizer)
    return {
        "Extracted Entities": entities,
        "Sentiment": sentiment
    }

# Streamlit app
st.title("Review Analyzer: Entity Extraction and Sentiment Classification")

# Input for review
review = st.text_area("Enter the product review:")

if st.button("Analyze"):
    if review:
        st.subheader("Analysis Results")

        # Analyze the review for entities and sentiment
        analysis = analyze_review(review, nlp, matcher, model_3, tfidf_vectorizer)

        # Display extracted entities
        st.write("### Extracted Entities:")
        st.json(analysis["Extracted Entities"])

        # Display sentiment classification
        st.write("### Sentiment Classification:")
        st.write(f"Sentiment: {analysis['Sentiment']}")
    else:
        st.error("Please enter a review to analyze.")
