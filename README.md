# SMS Spam Classification using Machine Learning

## Project Overview

This project focuses on classifying SMS messages as **Spam** or **Ham (legitimate messages)** using Natural Language Processing (NLP) and Machine Learning techniques.

The project includes exploratory data analysis, text feature extraction using `CountVectorizer`, and classification using multiple machine learning algorithms.

The following models are implemented and compared:

- Multinomial Naive Bayes
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

The performance of each model is evaluated using the test dataset.

---

## Objectives

The main objectives of this project are:

- Understand the structure of the SMS dataset.
- Perform Exploratory Data Analysis (EDA).
- Analyze the distribution of Spam and Ham messages.
- Visualize the distribution of message categories.
- Analyze frequently occurring words in SMS messages.
- Convert text messages into numerical features using `CountVectorizer`.
- Train multiple machine learning classification models.
- Compare the performance of different models.
- Identify the best-performing model for SMS spam classification.

---

## Dataset

The project uses the `spam.csv` dataset.

The dataset contains **5,572 SMS messages** with the following columns:

| Column | Description |
|--------|-------------|
| `Category` | Indicates whether the message is `ham` or `spam` |
| `Message` | Contains the actual SMS message |

### Dataset Distribution

| Category | Number of Messages |
|----------|-------------------:|
| Ham | 4,825 |
| Spam | 747 |
| **Total** | **5,572** |

The dataset contains significantly more legitimate messages than spam messages.

---

## Exploratory Data Analysis

The notebook performs several exploratory analysis steps, including:

- Checking the dimensions of the dataset.
- Checking the number of rows and columns.
- Inspecting column names.
- Finding unique categories.
- Calculating the frequency of each category.
- Visualizing the category distribution.
- Generating a word cloud to identify frequently occurring words.

### Categories

The dataset contains two categories:

- `ham`
- `spam`

### Class Distribution

The dataset contains:

- **4,825 Ham messages**
- **747 Spam messages**

This indicates that the dataset is imbalanced toward legitimate SMS messages.

---

## Data Visualization

The project includes visualizations to better understand the dataset.

### 1. Category Distribution

A visualization is created to show the distribution of:

- Ham messages
- Spam messages

### 2. Word Cloud

A word cloud is generated to visualize frequently occurring words in the SMS messages.

These visualizations help in understanding the characteristics and distribution of the text data.

---

## Feature Extraction

Machine learning algorithms require numerical input, while SMS messages are textual data.

To convert the messages into numerical features, the project uses:

```python
from sklearn.feature_extraction.text import CountVectorizer