# ML Ramp-Up — Week 1
**Asmicore Semiconductor | Academic Engagement Program | ML Specialization Track**

**Author:** Parizat Mitra Partha  
**Group:** C — UCI Heart Disease Dataset  
**Task:** Binary Classification — Heart Disease Prediction

---

## Dataset
- **Name:** UCI Heart Disease (Cleveland)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- **Samples:** 297 patients (after cleaning)
- **Features:** 13 original → 27 after engineering
- **Target:** `condition` — 0 = No Disease, 1 = Heart Disease

---

## Project Structure

```
├── data/
│   └── heart_cleveland_upload.csv       
├── notebooks/
│   └── week1.ipynb                 
├── outputs/
│   ├── day1.json
│   ├── day2.json
│   ├── day3.json
│   ├── day4.json
│   ├── day5.json
│   ├── day6.json
│   └── plots/                          
├── models/
│   ├── pipeline_logisticregression.pkl
│   ├── pipeline_decisiontree.pkl
│   ├── pipeline_randomforest.pkl
│   ├── pipeline_knn.pkl
│   └── pipeline_svm.pkl
├── logs/
│   └── issue_log.txt
├── src/
│   └── (processing scripts)
└── README.md
```

---

## Day-by-Day Work

| Day | Topic | Output |
|-----|-------|--------|
| 1 | Data Understanding & Repository Setup | `day1.json` |
| 2 | Exploratory Data Analysis | `day2.json` + 5 plots |
| 3 | Feature Engineering | `day3.json` + scaling plot |
| 4 | Model Training & Prediction Logging | `day4.json` + 2 plots |
| 5 | Evaluation & Performance Metrics | `day5.json` + 5 plots |
| 6 | Final Integration & Pipeline | `day6.json` + 5 .pkl files |

---

## Models Trained

| Model | Test Accuracy | Recall | F1 | AUC |
|-------|--------------|--------|----|-----|
| Logistic Regression | 0.9000 | 0.8571 | 0.8889 | 0.9676 |
| Decision Tree | 0.8333 | 0.7143 | 0.7571 | 0.8365 |
| Random Forest | 0.8500 | 0.7143 | 0.7857 | 0.9464 |
| **KNN ★** | **0.9000** | **0.8214** | **0.8519** | **0.9559** |
| SVM | 0.8833 | 0.7500 | 0.8077 | 0.9286 |

**Recommended model: KNN** — best F1 (0.8519) and best recall (0.8214).  
Recall is the priority metric — missing a disease case (False Negative) is the most dangerous outcome.

---

## Key Engineering Decisions

- **Scaler:** StandardScaler — best for LR/SVM with near-normal features
- **Encoding:** One-Hot Encoding for `cp`, `restecg`, `slope`, `thal`
- **Outliers:** IQR Winsorization on `chol` and `trestbps` — no rows removed
- **Leakage prevention:** Split first → fit scaler on X_train only → sklearn Pipeline enforces this in CV
- **Validation:** 5-Fold Stratified Cross-Validation on all pipelines

---

## Issues Encountered & Resolved

| # | Issue | Fix |
|---|-------|-----|
| 1 | Wrong target column name (`target` vs `condition`) | Used `TARGET_COLUMN` variable throughout |
| 2 | Windows path SyntaxError | Used raw string `r'C:\Users\...'` |
| 3 | Seaborn palette dict mismatch | Changed to list palette `["green", "red"]` |
| 4 | `logs/` folder not found | Added `os.makedirs('../logs', exist_ok=True)` |
| 5 | Data leakage risk | sklearn Pipeline — scaler fit per fold automatically |

---

## How to Run

```bash
# 1. Clone the repo
git clone git clone https://github.com/ParthaParizat/uci-heart-disease-.git
cd ml-ramp-up-week1

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn joblib

# 3. Download the dataset from Kaggle and place it in data/
#    https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

# 4. Open the notebook
jupyter notebook notebooks/week1.ipynb

# 5. Update DATA_PATH in Cell 1 to your local path
DATA_PATH = r"../data/heart_cleveland_upload.csv"
```

---

## Environment

- Python 3.10
- pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- Jupyter Notebook / Anaconda
