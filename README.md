# 📰 Fake News Detection using Machine Learning


https://github.com/user-attachments/assets/1296231f-7015-4b7f-ace9-47d03baa73b4


This project is a **Fake News Detection System** that classifies news articles as **real or fake** using a trained **RandomForestClassifier**. It includes:
- A Jupyter Notebook for development and experimentation(`fake_news_detector.ipynb`)
- A Python script for building a Streamlit web app(`app.py`)
- A trained model (`fakenews_detection_model.pkl`)
- A TF-IDF vectorizer (`vectorizer.pkl`)

## 💡 Project Overview

In today’s digital era, misinformation spreads fast and wide. This Fake News Detector helps identify whether a given news article is trustworthy or fabricated using natural language processing and machine learning.

The core idea is to allow users to input any news text and instantly know whether it's **real** or **fake**, through a user-friendly Streamlit interface.

---

## 📁 Project Structure

```
📦 Fake-News-Detection/
├── FakeNewsDetection.ipynb        # Jupyter notebook for EDA, preprocessing, model training
├── app.py                         # Streamlit app code
├── fakenews_detection_model.pkl   # Trained RandomForest model
├── vectorizer.pkl                 # Fitted TfidfVectorizer
├── requirements.txt               # Required libraries
└── README.md                      # Project documentation
```

---

## 🚀 How to Run

### 🔧 Installation

```bash
git clone https://github.com/najmarazzaq761/Fake-News-Detector-Using-ML.git
cd Fake-News-Detector-Using-ML
pip install -r requirements.txt
```

### ▶️ Launch the App

```bash
streamlit run app.py
```

You will see a text input box where you can paste a news article. Hit the **Predict** button to classify it as *Real* or *Fake*.

---

## 🛠 Features

- TF-IDF based feature extraction
- Stopwords removal, lemmatization, and text cleaning
- Machine learning model (Random Forest)
- Interactive web app using Streamlit

---

## 🧠 Model Details

- **Algorithm Used**: RandomForestClassifier
- **Vectorization**: TF-IDF (`max_features=5000`)
- **Accuracy Achieved**: _Your model accuracy here_ (add from notebook)

---

## 📊 Exploratory Data Analysis (EDA)

The notebook includes:
- Word clouds for fake vs real news
- Text length distributions
- Most frequent words
- Data cleaning & preprocessing steps

---

## 🖼️ Streamlit App UI

- User inputs news content in a text field
- Behind the scenes, the app cleans and vectorizes the input
- Prediction is made using the trained model
- Real-time results shown to the user

---

## 📚 Requirements

```
streamlit
nltk
scikit-learn
matplotlib
seaborn
pandas
wordcloud
```

Install them via:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

**Najma Razzaq**  
BSCS Student | Data Scientist | [LinkedIn](https://www.linkedin.com/in/najmarazzaq)

---

## 📌 License

This project is licensed under the MIT License.

```
