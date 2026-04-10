import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import the cleaner function from your preprocessing script
# We only import the function; it won't run the whole file because of the `if __name__ == "__main__":` block!
from data_preprocessing import preprocess_text

if __name__ == "__main__":
    cache_file = "cleaned_imdb_dataset.csv"
    
    # ----------------------------------------
    # 1. Load Data (With caching for speed)
    # ----------------------------------------
    if os.path.exists(cache_file):
        print(f"\n--- Loading cached preprocessed data from '{cache_file}' ---")
        df = pd.read_csv(cache_file)
        
        # Drop any empty reviews that might have occurred after removing all special chars
        df = df.dropna(subset=['cleaned_review']) 
    else:
        print("\n--- Loading raw dataset and running preprocessing ---")
        print("(This will take a minute, but we'll save it so it's instant next time!)")
        
        df = pd.read_csv("IMDB_Dataset.csv")
        df['cleaned_review'] = df['review'].apply(preprocess_text)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Save to CSV for future runs
        df.to_csv(cache_file, index=False)
        print(f"-> Saved preprocessed data to '{cache_file}'")

    # ----------------------------------------
    # 2. Data Splitting
    # ----------------------------------------
    print("\n--- Splitting Dataset ---")
    X = df['cleaned_review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    # ----------------------------------------
    # 3. TF-IDF Vectorization
    # ----------------------------------------
    print("\n--- Applying TF-IDF Vectorization ---")
    tfidf = TfidfVectorizer(max_features=5000) 
    
    # Fit & Transform on Training Data
    X_train_tfidf = tfidf.fit_transform(X_train)
    # Transform ONLY on Test Data
    X_test_tfidf = tfidf.transform(X_test)
    
    print("TF-IDF Vectorization Complete!")

    # ----------------------------------------
    # 4. Model Training: Naive Bayes
    # ----------------------------------------
    print("\n--- Training Naive Bayes Model ---")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    print("Model training complete.")
    
    # ----------------------------------------
    # 5. Making Predictions & Evaluation
    # ----------------------------------------
    print("\n--- Evaluating the Naive Bayes Model ---")
    y_pred = nb_model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f} (Percentage of total correct predictions)")
    print(f"Precision: {precision:.4f} (When predicting positive, how often is it right?)")
    print(f"Recall:    {recall:.4f} (Out of all actual positives, how many did it find?)")
    print(f"F1-Score:  {f1:.4f} (Harmonic mean of Precision and Recall)")
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print("                   Predicted Negative  Predicted Positive")
    print(f"Actual Negative |  {cm[0][0]:<17} | {cm[0][1]}")
    print(f"Actual Positive |  {cm[1][0]:<17} | {cm[1][1]}")

    # ----------------------------------------
    # 6. Model Training: Logistic Regression
    # ----------------------------------------
    print("\n--- Training Logistic Regression Model ---")
    # max_iter=1000 ensures the algorithm has enough passes to converge mathematically without throwing a warning
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train_tfidf, y_train)
    print("Model training complete.")
    
    # ----------------------------------------
    # 7. Evaluating the Logistic Regression
    # ----------------------------------------
    print("\n--- Evaluating the Logistic Regression Model ---")
    y_pred_lr = logreg_model.predict(X_test_tfidf)
    
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    precision_lr = precision_score(y_test, y_pred_lr)
    recall_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    
    print(f"Accuracy:  {accuracy_lr:.4f} (Percentage of total correct predictions)")
    print(f"Precision: {precision_lr:.4f} (When predicting positive, how often is it right?)")
    print(f"Recall:    {recall_lr:.4f} (Out of all actual positives, how many did it find?)")
    print(f"F1-Score:  {f1_lr:.4f} (Harmonic mean of Precision and Recall)")
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    
    print("\nConfusion Matrix:")
    print("                   Predicted Negative  Predicted Positive")
    print(f"Actual Negative |  {cm_lr[0][0]:<17} | {cm_lr[0][1]}")
    print(f"Actual Positive |  {cm_lr[1][0]:<17} | {cm_lr[1][1]}")

    print("\n--- Pipeline Completed Successfully ---")
