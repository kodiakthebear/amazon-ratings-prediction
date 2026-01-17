# Amazon Product Rating Prediction System 

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A Natural Language Processing (NLP) project that classifies E-commerce product reviews into numeric ratings (1-5 stars). This project compares traditional Machine Learning algorithms against Deep Learning architectures, utilizing advanced text representation techniques like TF-IDF and Pre-trained Word2Vec embeddings.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Analysis](#dataset-analysis)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [Model Architectures](#model-architectures)
- [Performance Results](#performance-results)
- [Installation & Usage](#installation--usage)
- [Future Improvements](#future-improvements)

## 🔍 Project Overview
The goal of this project is to predict the sentiment score (1-5) of a product based solely on the textual content of its review. The project addresses the challenges of multi-class text classification and severe class imbalance using various architectural approaches.

## 📊 Dataset Analysis
The dataset consists of **59,622** Amazon product reviews.
* **Columns:** `Score` (1-5) and `Text` (Review content).
* **Cleaning:** Missing values were imputed (median for scores, placeholders for text) and 14,000+ duplicates were removed.
* **Distribution:** Exploratory Data Analysis (EDA) revealed a significant **Class Imbalance**, with 5-star reviews heavily dominating the dataset (>30k instances), while 1-4 star reviews combined comprised <25k instances.

## ⚙️ Methodology

### Preprocessing
To ensure high-quality input data, a rigorous cleaning pipeline was implemented using **NLTK**:
1.  **Normalization:** Conversion to lowercase.
2.  **Noise Removal:** Stripping of punctuation and stopwords.
3.  **Tokenization:** Splitting sentences into individual words.
4.  **Lemmatization:** Reducing words to their root form (e.g., "buying" $\rightarrow$ "buy") using WordNet to reduce dimensionality.

### Text Representation
* **TF-IDF:** Used for Naïve Bayes and KNN to reflect word importance.
* **Word2Vec (Google News 300):** Used for CNN and LSTM layers to capture semantic relationships between words.

### Model Architectures
1.  **Naïve Bayes (MultinomialNB):** A probabilistic baseline model using TF-IDF vectors.
2.  **K-Nearest Neighbors (KNN):** Instance-based learning using Cosine Similarity. Hyperparameter tuning identified optimal $k=9$.
3.  **Convolutional Neural Network (CNN):**
    * Layers: Embedding (frozen Word2Vec) $\rightarrow$ Conv1D (128 filters) $\rightarrow$ Conv1D (64 filters) $\rightarrow$ GlobalMaxPooling $\rightarrow$ Dense (Softmax).
    * Designed to capture local n-gram patterns.
4.  **Long Short-Term Memory (LSTM):**
    * Layers: Embedding (frozen Word2Vec) $\rightarrow$ Bidirectional LSTM $\rightarrow$ Dropout $\rightarrow$ Dense (Softmax).
    * Designed to capture long-term dependencies and sequential context.

## 📈 Performance Results

The models were evaluated on Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM (Bidirectional)** | **73.30%** | **0.71** | **0.72** | **0.72** |
| **CNN** | 72.43% | 0.71 | 0.72 | 0.70 |
| **KNN (k=9)** | 67.24% | 0.52 | 0.40 | 0.44 |
| **Naïve Bayes** | 67.20% | 0.64 | 0.35 | 0.38 |

> **Insight:** The LSTM outperformed other models by effectively capturing context via bidirectional learning. Traditional ML models (KNN/NB) struggled significantly with the class imbalance, often failing to distinguish minority classes effectively compared to the Neural Networks.

## 💻 Installation & Usage

### Prerequisites
* Python 3.x
* Jupyter Notebook

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/amazon-rating-prediction.git](https://github.com/yourusername/amazon-rating-prediction.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk gensim
    ```
3.  **Important:** Ensure you have the Google News Word2Vec model available or allow `gensim.downloader` to fetch it (approx 1.6GB).

### Predicting a Rating
The notebook includes a standalone function `predict_product_rating`. To use it:

1.  Run the notebook to train and save the models as `.pkl` files.
2.  Execute the interaction cell:
    ```python
    text = "This product is decent. It is alright."
    model = "LSTM" # Options: NBModel, KNNmodel, CNN, LSTM
    predict_product_rating(text, model)
    ```

## 🚀 Future Improvements
* **Handling Imbalance:** Implement SMOTE or Class Weighting to penalize misclassification of minority classes (1-3 stars).
* **Advanced Embeddings:** Fine-tune BERT or RoBERTa transformers for better contextual understanding.
* **Hyperparameter Tuning:** Use KerasTuner to optimize CNN filter sizes and LSTM units.

## 📜 License
This project is open-source and available under the MIT License.
