# 🎧 Spotify Skip Prediction using Logistic Regression

## 📌 Project Overview

This project uses **Logistic Regression** to predict whether a user will skip a song within the first 30 seconds, based on audio features such as danceability, energy, loudness, tempo, and more.

The dataset was synthetically generated to simulate real-world music streaming behavior, with a binary target variable:

* `1` → Skipped
* `0` → Not Skipped

---

## 🧠 Problem Statement

With millions of tracks streamed daily, understanding **user skip behavior** is key to enhancing recommendation systems and listener experience.

> Goal: **Classify songs as "skipped" or "not skipped"** using audio features.

---

## 📂 Dataset

* **Samples:** 1000
* **Features:**

  * `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `valence`, `tempo`
* **Target:** `skip_30s` (0 = Not Skipped, 1 = Skipped)

📝 Dataset generated using NumPy & Pandas to reflect realistic ranges.

---

## 🧪 Approach

1. **Data Exploration**: Class balance, correlations, distributions
2. **Preprocessing**: Feature scaling with `StandardScaler`
3. **Model Training**: `LogisticRegression` from Scikit-Learn
4. **Evaluation**:

   * Accuracy
   * Confusion Matrix
   * Precision, Recall, F1-score

---

## 📈 Results

* **Accuracy:** \~88% (may vary)
* **F1-score for skipped songs:** Indicates model's ability to detect actual skips despite class imbalance
* **Confusion Matrix**: Gives insight into model strengths & weaknesses

---

## 💡 Learnings

* Logistic Regression performs surprisingly well on binary classification with properly scaled features.
* Accuracy alone is **not** sufficient on imbalanced datasets — precision, recall, and F1-score provide deeper insights.
* Feature importance via model coefficients can offer interpretability.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy, Seaborn, Matplotlib
* Scikit-Learn (for model and metrics)

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies
3. Run `main.py` or `notebook.ipynb`
4. View outputs and plots

---

## 📚 Future Improvements

* Try other models: Decision Trees, Random Forest, XGBoost
* Handle imbalance using SMOTE or class weights
* Use a real-world dataset (e.g., from Spotify’s MSSD)
* Add hyperparameter tuning

---
