# Fake News Detection Using Machine Learning and Explainable AI

## Project Overview

This project is a machine learning application that classifies news articles as either **Fake News** or **Real News**.

The system uses text classification techniques to analyze news content and predict its authenticity based on patterns learned from the training data. It also uses SHAP (SHapley Additive exPlanations) to explain the predictions by identifying words that influence the model's decision.

The application is developed using Python and Gradio, providing a simple web interface where users can select sample articles, choose a vectorization technique, and view predictions from two machine learning models.

## Objectives

- Develop a machine learning system for fake news detection.
- Compare Logistic Regression and Random Forest classifiers.
- Apply Bag-of-Words and TF-IDF for text representation.
- Use SHAP to explain model predictions.
- Develop an interactive web application using Gradio.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas and NumPy | Data processing and manipulation |
| Scikit-learn | Text vectorization and machine learning models |
| NLTK | Text preprocessing |
| SHAP | Explaining model predictions |
| Gradio | Creating the web application |
| Jupyter Notebook | Model development and experimentation |

## Machine Learning Models

The project uses two classification algorithms:

**1. Logistic Regression**

A classification algorithm used to predict whether a news article is fake or real based on its text features.

**2. Random Forest**

An ensemble learning algorithm that combines multiple decision trees to classify news articles.

Both models are evaluated using two text vectorization techniques:

- **Bag-of-Words (BoW):** Represents text using word occurrence counts.
- **TF-IDF:** Represents text using word importance based on its frequency across documents.

This allows the performance of both models to be compared using different text representations.

## Text Preprocessing

The news articles are prepared before being used by the machine learning models. The preprocessing steps include:

- Expanding contractions.
- Tokenizing the text.
- Removing punctuation.
- Lemmatizing words.

The processed text is then converted into numerical features using the selected vectorization technique.

## Explainable AI (SHAP)

SHAP is used to explain the predictions made by the trained models.

For each article, the application displays words that contribute toward the Fake News prediction and words that contribute toward the Real News prediction.

These explanations help users understand which words influenced the model's decision.

SHAP explains the model's behavior and does not verify whether the article is factually correct.

## Application Features

- Select from four synthetic news articles.
- Choose between Bag-of-Words and TF-IDF.
- View predictions from Logistic Regression and Random Forest.
- View SHAP explanations for both models.
- Compare how different models classify the same article.

## Project Structure

```text
fake-news-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── artifacts/
│   ├── lr_tfidf.pkl
│   ├── rf_tfidf.pkl
│   ├── lr_bow.pkl
│   ├── rf_bow.pkl
│   ├── tfidf.pkl
│   └── bow.pkl
│
└── notebook/
    └── fake_news_detection.ipynb
```

The `artifacts` folder contains the trained models and vectorizers used by the application.

The notebook contains the model development, training, and evaluation process.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python app.py
```

The Gradio application will start locally. Open the local URL displayed in the terminal to access the interface.

## Deployment

The application can be deployed using a Python hosting platform such as Render or Hugging Face Spaces.

The trained models and vectorizers must be included in the deployment environment so the application can generate predictions without retraining the models.

The source code is available on GitHub.

**Live Application:** Add your deployed application URL here.

## Limitations

- Predictions depend on patterns learned from the training dataset.
- The models may not perform well on news articles that differ significantly from the training data.
- SHAP explanations show how words influence the model's prediction but do not establish whether the information is true or false.
- The application uses synthetic articles for demonstration and does not independently verify news sources.

## Disclaimer

This project is developed for educational purposes. Its predictions should not be treated as definitive verification of news authenticity.
