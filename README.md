# IMDB Sentiment Analysis using Machine Learning

#Overview
This project aims to classify IMDB movie reviews as **positive** or **negative** using machine learning models.  

The goal is to compare the performance of two algorithms:
- Multinomial Naive Bayes
- Logistic Regression

---
# Repositry structure
├── data_preprocessing.py
├── train_models.py
├── cleaned_imdb_dataset.csv
├── README.md

# Libraries need to reproduce findings
Panda, Numpy, Matplotlib, Seaborn, joblib, scikit-learn.

# Research Question
How accurately can machine learning models classify IMDB movie reviews as positive or negative, and which model performs best?

---
Dataset
- Source: IMDB Movie Reviews Dataset  
- Total Samples: 50,000 reviews  
- Features:
  - `review` (text)
  - `sentiment` (positive/negative)  
- Balanced dataset (25,000 positive, 25,000 negative)

---

## How to reproduce findings

### 1. Data Preprocessing
### 2. Models Used
### 3. Evaluation Metrics
---
# Key finding
Logistic Regression outperformed Naive Bayes across all metrics.
