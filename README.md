# Probabilistic SMS Spam Detection

An SMS spam classifier that bridges **probability theory** with a practical **machine learning pipeline** — built as part of the BAMAT207 (Probability and Statistics) coursework. The project doesn't just classify messages as Spam or Ham; it mathematically demonstrates *why* the classification works, using prior probability, conditional probability, Bayes' Theorem, and random variable analysis, all verified against a real trained model.

---

## 📖 Project Overview

This project uses the **UCI SMS Spam Collection Dataset** (5,572 labelled messages) to:

1. Analyze spam/ham class distribution and message patterns through EDA
2. Calculate prior and conditional probabilities for suspicious words, and verify them using Bayes' Theorem
3. Model spam classification as a **Bernoulli random variable** and compute its Expectation and Variance
4. Train a **Multinomial Naive Bayes** classifier on vectorized message text
5. Evaluate the model with standard classification metrics
6. Provide real-time predictions with probability scores for any new message

The goal is to show, end-to-end, how the probability concepts taught in BAMAT207 directly power a real, working AI system — not just as separate theory, but as the literal mechanism behind the model's predictions.

---

## ✨ Features

- ✅ Automated dataset loading directly from UCI (no manual downloads)
- ✅ Full data cleaning pipeline (missing values, duplicates, label encoding)
- ✅ Exploratory Data Analysis with 6+ visualizations (bar chart, pie chart, histograms, box plot, word frequency)
- ✅ Manual probability calculations for 5 suspicious words, **independently verified using Bayes' Theorem**
- ✅ Bernoulli random variable analysis (Expectation & Variance) matching theoretical formulas exactly
- ✅ Stratified train-test split preserving class balance
- ✅ Multinomial Naive Bayes model with Laplace smoothing
- ✅ Full evaluation suite: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- ✅ Real-time `predict_sms()` function with spam/ham probability output
- ✅ Saved, reusable model and vectorizer (`.pkl` files via Joblib)

---

## 🛠️ Technologies & Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Probability, expectation, and variance calculations |
| `matplotlib` | All visualizations (charts, histograms, heatmap) |
| `scikit-learn` | Train-test split, CountVectorizer, Naive Bayes, evaluation metrics |
| `re` | Word-level text search for probability analysis |
| `joblib` | Saving/loading the trained model and vectorizer |
| `requests` / `zipfile` | Direct dataset download from UCI |

**Platform:** Google Colab (development) · GitHub (version control & hosting)

---

## ⚙️ Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/Barath0303/probabilistic-sms-spam-detection.git
cd probabilistic-sms-spam-detection
pip install pandas numpy matplotlib scikit-learn joblib requests
```

If running in Google Colab, only one extra install is needed (everything else is pre-installed):

```python
!pip install joblib
```

---

## 🚀 Usage

### Run the full notebook
Open `SMS_Spam_Detection.ipynb` in Google Colab or Jupyter and run all cells top to bottom — each phase builds on the previous one's output.

### Use the trained model directly
```python
import joblib

model = joblib.load("sms_spam_detection_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sms(message: str) -> dict:
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    proba = model.predict_proba(message_vec)[0]
    return {
        "message": message,
        "prediction": "Spam" if prediction == 1 else "Ham",
        "P(Spam)": round(proba[1], 4),
        "P(Ham)": round(proba[0], 4)
    }

result = predict_sms("Free entry into our contest.")
print(result)
```

**Example output:**
```
{'message': 'Free entry into our contest.', 'prediction': 'Spam', 'P(Spam)': 0.9926, 'P(Ham)': 0.0074}
```

---

## 📊 Results

### Dataset Summary
- **Total messages (raw):** 5,572
- **Duplicates removed:** 403
- **Final clean dataset:** 5,169 messages
- **Class distribution:** 87.4% Ham (4,516) · 12.6% Spam (653)

### Probability Analysis — Conditional Probabilities (Direct = Bayes' Theorem, both verified identical)

| Word | Messages containing word | Of which Spam | P(Spam \| word) |
|---|---|---|---|
| prize | 73 | 73 | **1.0000** |
| urgent | 62 | 57 | **0.9194** |
| win | 55 | 49 | **0.8909** |
| offer | 30 | 24 | **0.8000** |
| free | 203 | 148 | **0.7291** |

Prior probability: **P(Spam) = 0.1263**, **P(Ham) = 0.8737**

### Random Variable Analysis
Modeling spam as a Bernoulli random variable Y (1 = Spam, 0 = Ham):
- **E(Y) = 0.1263** (exactly matches P(Spam), as theory predicts)
- **Var(Y) = 0.1104** (formula p(1−p) matches pandas' computed variance exactly)
- **SD(Y) = 0.3322**

### Model Performance (on unseen test set, 1,034 messages)

| Metric | Score |
|---|---|
| Accuracy | **98.16%** |
| Precision | **95.90%** |
| Recall | **89.31%** |
| F1-Score | **92.49%** |

**Confusion Matrix:**

| | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 898 (TN) | 5 (FP) |
| **Actual Spam** | 14 (FN) | 117 (TP) |

Only **5 genuine messages** (0.55%) were wrongly flagged as spam, while **14 spam messages** (10.69%) slipped through undetected — a deliberately conservative trade-off, since falsely blocking a real message is a worse outcome than letting some spam through.

### Sample Real-Time Predictions

| Message | Prediction | P(Spam) |
|---|---|---|
| "Congratulations! You won a free prize. Claim now." | Spam | 1.0000 |
| "Free entry into our contest." | Spam | 0.9926 |
| "URGENT! Your account has been suspended, click here to verify." | Spam | 0.9970 |
| "Hey, are we still meeting for lunch tomorrow?" | Ham | 0.0000 |
| "Can you pick up some milk on your way home?" | Ham | 0.0000 |

---

## 🔮 Future Improvements

- Replace `CountVectorizer` with `TfidfVectorizer` to down-weight common but low-information words
- Experiment with stop-word removal and stemming/lemmatization
- Compare Naive Bayes against Logistic Regression or SVM
- Deploy as a simple web app (Streamlit/Flask) for interactive use
- Expand suspicious-word analysis beyond the 5 initial words using automated feature importance ranking

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository — SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- Developed as part of **BAMAT207 – Probability and Statistics** coursework
