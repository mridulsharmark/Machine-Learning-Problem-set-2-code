# ============================================
# IMDB Sentiment Analysis - Data Preprocessing
# ============================================

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# --------------------------------------------
# Text Preprocessing Function
# --------------------------------------------
def preprocess_text(text):
    """
    Cleans the input text by:
    - Converting to lowercase
    - Removing HTML tags
    - Removing punctuation and numbers
    - Removing stopwords
    """

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags (e.g., <br />)
    text = re.sub(r'<.*?>', ' ', text)

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]

    return " ".join(words)


# --------------------------------------------
# Main Execution
# --------------------------------------------
if __name__ == "__main__":

    print("\n--- Loading Dataset ---")

    # Load dataset
    df = pd.read_csv("IMDB_Dataset.csv")

    # Basic info
    print("\nDataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\nSample Data:")
    print(df.head(3))

    # ----------------------------------------
    # Preprocessing
    # ----------------------------------------
    print("\n--- Starting Text Preprocessing ---")
    print("(Processing 50,000 reviews... please wait)")

    df['cleaned_review'] = df['review'].apply(preprocess_text)

    print("\nSample After Cleaning:")
    print(df[['review', 'cleaned_review', 'sentiment']].head(3))

    # ----------------------------------------
    # Encode Target Labels
    # ----------------------------------------
    print("\n--- Encoding Sentiment Labels ---")

    # Convert 'positive' -> 1 and 'negative' -> 0
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    print(df['sentiment'].value_counts())

    # ----------------------------------------
    # Train-Test Split
    # ----------------------------------------
    print("\n--- Splitting Dataset ---")

    X = df['cleaned_review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    print("\nData Split Complete!")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # ----------------------------------------
    # TF-IDF Vectorization
    # ----------------------------------------
    print("\n--- Applying TF-IDF Vectorization ---")
    
    # Initialize the TF-IDF Vectorizer
    # We set max_features to 5000 to keep the most important words and save memory/computation
    tfidf = TfidfVectorizer(max_features=5000) 
    
    # Fit the vectorizer on the training data ONLY, then transform the training data
    # We never 'fit' on the test data to prevent data leakage!
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # Transform the test data using the vocabulary learned from the training data
    X_test_tfidf = tfidf.transform(X_test)
    
    print("TF-IDF Vectorization Complete!")
    print(f"X_train_tfidf shape (samples, features): {X_train_tfidf.shape}")
    print(f"X_test_tfidf shape (samples, features): {X_test_tfidf.shape}")

    print("\n--- Preprocessing Completed Successfully ---")