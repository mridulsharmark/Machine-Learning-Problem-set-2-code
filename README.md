# IMDB Sentiment Analysis using Machine Learning

# Overview
This project aims to classify IMDB movie reviews as **positive** or **negative** using machine learning models.  

The goal is to compare the performance of two algorithms:
- Multinomial Naive Bayes
- Logistic Regression

---
# Repository structure

'data_preprocessing.py'
'train_models.py'
'README.md'

# Libraries need to reproduce findings
Pandas, Numpy, Matplotlib, Seaborn, joblib, scikit-learn.

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

## Steps to reproduce finding.

### 1. Install required liabraries
### 2. Download dataset from the link given below
### 3. Data preprocessing
### 4. Train and evaluate models
---
# Key finding.
Logistic Regression outperformed Naive Bayes across all metrics.

Dataset- https://www.kaggle.com/datasets/rehanliaqat17/imbd-dataset

