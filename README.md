# 🧠 AutoML Document Analyzer

A production-ready, end-to-end **AutoML** web application built with **Streamlit** and **scikit-learn**.  
Upload any CSV or Excel dataset and the system automatically performs EDA, preprocessing, model training, evaluation, and lets you export results — all through a polished, interactive UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📂 **File Upload** | CSV, XLSX, XLS — graceful handling of malformed / missing data |
| 🔍 **Automated EDA** | Overview stats, column types, missing-value analysis, correlation heatmap, distributions, categorical counts |
| 🎯 **Problem Detection** | Automatically classifies task as **Classification** or **Regression** based on target column |
| ⚙️ **Preprocessing Pipeline** | Median/mode imputation, OHE (low cardinality), label encoding (high cardinality), StandardScaler |
| 🤖 **Multi-Model Training** | Logistic Regression + Random Forest (classification) · Linear Regression + Random Forest (regression) |
| 📊 **Rich Evaluation** | Accuracy, Precision, Recall, F1 · Confusion matrices · RMSE, R² · Feature importance charts |
| 💾 **Export** | Download best model (`.pkl`), full report (`.json`), metrics table (`.csv`) |

---

## 🚀 Quick Start

### 1 — Clone & install

```bash
git clone https://github.com/your-username/automl-document-analyzer.git
cd automl-document-analyzer
pip install -r requirements.txt
```

### 2 — Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
automl_document_analyzer/
│
├── app.py              # Streamlit UI — main entry point
├── eda.py              # Exploratory Data Analysis module
├── preprocess.py       # Preprocessing pipeline (Preprocessor class)
├── model.py            # Model training, evaluation, visualisation
│
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🔄 Pipeline Overview

```
Upload CSV/Excel
      │
      ▼
Exploratory Data Analysis
 ├── Shape, dtypes, duplicates
 ├── Missing value analysis
 ├── Descriptive statistics
 └── Visualisations (heatmap, distributions, value counts)
      │
      ▼
Preprocessing
 ├── Drop high-missing columns (> 80 %)
 ├── Median / mode imputation
 ├── One-hot encode low-cardinality cats
 ├── Label-encode high-cardinality cats
 └── StandardScaler → Train-test split (80/20)
      │
      ▼
Model Training
 ├── Classification: Logistic Regression + Random Forest
 └── Regression:     Linear Regression  + Random Forest
      │
      ▼
Evaluation & Selection
 ├── Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
 ├── Regression:     RMSE, R², Actual-vs-Predicted scatter
 └── Best model auto-selected by F1 / R²
      │
      ▼
Export
 ├── best_model.pkl (joblib)
 ├── automl_report.json
 └── model_metrics.csv
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web UI |
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, models, metrics |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `openpyxl` / `xlrd` | Excel file support |
| `joblib` | Model serialisation |

---

## 🧪 Sample Datasets to Try

- **Classification**: [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv), [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- **Regression**: [Boston Housing](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv), [California Housing](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv)

---

## 🤝 Contributing

Pull requests welcome!  
Open an issue first to discuss significant changes.

---

## 📄 License

MIT © 2024 — Free to use and modify.
