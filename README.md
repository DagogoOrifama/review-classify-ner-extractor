---
title: DS TF NER
emoji: 👁
colorFrom: green
colorTo: red
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
short_description: A review classification and NER extraction project
---

# opinion-minning-classifier-ner-extractor
Classify sentiments on reviews and extract entities

# Review Analyzer: Entity Extraction and Sentiment Classification

## Overview
This project implements a solution for analyzing product reviews by extracting entities and classifying sentiments. The primary tasks include **Sentiment Classification** and **Named Entity Recognition (NER)**, with the goal of providing actionable insights from customer feedback. The project is hosted on **Hugging Face Spaces** and leverages state-of-the-art machine learning techniques to classify review sentiments and recognize entities like products and parts.

## Features
- **Sentiment Classification**: Classifies product reviews into Positive, Neutral, or Negative sentiments using a TensorFlow-based neural network.
- **Named Entity Recognition (NER)**: Extracts product-related entities (e.g., "tablet", "screen", "battery") using spaCy’s custom matcher.
- **Data Augmentation**: Utilizes the Pegasus model for paraphrasing to balance the dataset and improve model generalization.
- **Visualization**: Provides insights into review lengths, word distributions, and entity sentiments using graphs and heatmaps.

## Dataset
The project uses the [**Amazon Product Reviews**](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) dataset from Kaggle, specifically focusing on reviews of the **Fire Tablet, 7 Display, Wi-Fi, 16 GB - Includes Special Offers, Black**. Data preprocessing includes:
- Text cleaning (removing stopwords, punctuation).
- Tokenization and TF-IDF vectorization for feature extraction.
- Labeling review sentiments based on star ratings (1-2 as Negative, 4-5 as Positive, 3 as Neutral).

## Model Design
1. **Sentiment Classification**:
    - TensorFlow-based neural network with L2 regularization and dropout to prevent overfitting.
    - Three experiments were conducted using SMOTE and paraphrasing for data augmentation.
2. **Named Entity Recognition (NER)**:
    - Uses spaCy matcher to recognize product and part entities with custom rules for extracting common terms like "tablet", "camera", etc.

## Experiments and Results
- **Model 1**: Baseline with no regularization and dropout (using SMOTE for balancing).
- **Model 2**: Added regularization and dropout to improve generalization.
- **Model 3**: Further improvement with data augmentation (paraphrasing).
- Accuracy and performance were measured using metrics like Precision, Recall, and F1-Score. Visualized results with confusion matrices and accuracy curves.

## File Descriptions

- **app.py**:  
  The main script for the Streamlit app, which integrates the sentiment classification and named entity recognition models. It handles user inputs, performs predictions, and displays the results in an interactive web interface.

- **Experiments.ipynb**:  
  Jupyter notebook containing the three main experiments for sentiment classification and entity recognition models, along with other analyses, including the evaluation metrics and model performance results.

- **requirements.txt**:  
  A list of Python dependencies needed to run the project, including libraries like TensorFlow, spaCy, and Streamlit.

- **model_3.h5** and **tfidf_vectorizer.pkl**:  
  These files, containing the sentiment classification model and TF-IDF vectorizer, are stored on Hugging Face's model hub and are fetched dynamically by the app.

- **README.md**:  
  A detailed overview of the project, explaining its purpose, setup instructions, and usage.


## System Dependencies

In addition to the Python libraries listed in `requirements.txt`, the following system dependencies are required for this project:

- **libblas-dev**:  
  Provides the Basic Linear Algebra Subprograms (BLAS) library, which is essential for performing efficient matrix operations in numerical computations.

- **liblapack-dev**:  
  A library for linear algebra routines, used for solving systems of linear equations, linear least squares problems, and more.

- **gfortran**:  
  The GNU Fortran compiler, required for compiling scientific and mathematical libraries, particularly those that rely on Fortran-based linear algebra routines.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/review-classify-ner-extractor.git
    cd review-classify-ner-extractor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the project:
    ```bash
    streamlit run app.py
    ```

## Dependencies
- **TensorFlow**: For training the neural network model.
- **spaCy**: For performing named entity recognition.
- **Transformers**: For Pegasus paraphrasing model.
- **Streamlit**: For deploying the web application.
- **scikit-learn**: For data preprocessing and model evaluation.

## Usage

1. Run the app locally or visit the deployed version on Hugging Face Spaces.
2. Enter a product review in the text area and click "Analyze".
3. The app will display:
   - **Extracted Entities**: Key product-related entities.
   - **Sentiment Classification**: Positive, Neutral, or Negative sentiment of the review.

## Results Visualization
The project includes visualizations for:
- **Sentiment Distribution**: Review lengths and sentiment distributions.
- **Entity Recognition**: Co-occurrence matrix of products and parts.
- **Performance**: Confusion matrices and accuracy curves for model evaluation.

## Future Improvements
- **Enhancing Entity Recognition**: Add more comprehensive rules and machine learning models for better entity recognition.
- **Dataset Expansion**: Incorporate more reviews from different products for improved model generalization.
- **Advanced NLP Models**: Experiment with advanced transformer models like BERT for both sentiment analysis and NER.

## Acknowledgments
Special thanks to:
- Kaggle for the dataset.
- Hugging Face for providing model hosting and deployment services.
- 
## Author

This project was developed by **Dagogo Orifama**.  
For more information or inquiries, feel free to connect via:

- [GitHub](https://github.com/DagogoOrifama)
- [LinkedIn](https://www.linkedin.com/in/dagogoorifama/)
