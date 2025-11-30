# 🍽️ Recipe Rating Prediction — Machine Learning Project

Predicting recipe ratings (0–5 stars) from user reviews and metadata using NLP + Machine Learning.

---

## 📌 Project Overview
Recipe platforms often contain written reviews but not every review includes a star rating, which weakens personalization and recommendation quality.  
This project builds a model that **predicts the star rating purely from user review text and recipe-related features**, helping platforms:

- Auto-generate missing ratings  
- Improve recommendation systems  
- Detect biased or low-effort ratings  

---

## 📂 Dataset
| Source | Kaggle — *Recipe for Rating* dataset |
|--------|-------------------------------------|
| Training Samples | ~13,600 |
| Rating Classes | 0–5 stars |
| Data Types | Text + Numerical + Categorical |

Key features used:
`RecipeName`, `UserReputation`, `ReplyCount`, `ThumbsUpCount`, `ThumbsDownCount`, `Recipe_Review`

---

## 🧠 Methodology

### 🔹 Data Preprocessing
- Handled missing text values
- Dropped ID-type columns not useful for predictions
- Converted timestamp to datetime format
- Train-test split

### 🔹 NLP Processing
- Custom text cleaning (lowercase, removed links, punctuation)
- **TF–IDF Vectorization** (2000-dimensional feature space)

### 🔹 Numerical & Categorical Features
- One-Hot Encoding for `RecipeName`
- Standard Scaling for numeric variables

### 🔹 End-to-end ML Pipeline
All preprocessing + vectorization + scaling was integrated using **Scikit-learn’s `Pipeline` + `ColumnTransformer`** to ensure modularity and reproducibility.

---

## 🤖 Models Trained & Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline |
| Random Forest | Ensemble |
| Support Vector Machine | Non-linear |
| **HistGradientBoosting Classifier** | ⭐ Best performance |

### 🏆 Best Model Summary
| Metric | Value |
|--------|-------|
| Training Accuracy | 86% |
| **Test Accuracy** | **78% (Best)** |
| Generalization | Strong & stable across classes |

---

## 📊 Results & Visualization
- Target rating distribution
- Correlation heatmap
- Box plots for outlier exploration
- ROC curves for each class (One-vs-Rest)
- Training vs testing accuracy comparison

---

## 🧪 Test Predictions
Pipeline inference → Results exported as `submission.csv`:

```python
prediction = histclf.predict(transformed_test)
