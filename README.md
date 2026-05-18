# 🎓 College Dropout Risk Prediction

> An AI-powered web application that predicts student dropout risk using Machine Learning and provides personalised intervention recommendations.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square)
![Accuracy](https://img.shields.io/badge/Model%20R²-90.9%25-brightgreen?style=flat-square)

---

## 📌 Problem Statement

College dropout is a critical issue affecting students, institutions, and society. Early identification of at-risk students allows timely interventions — counselling, financial aid, academic support — that can significantly improve student retention.

This project builds a **regression model** that predicts a continuous **dropout risk score** (0 = safe, 1 = high risk) based on a student's academic and personal profile.

---

## 🚀 Live Demo

```bash
streamlit run app.py
```

---

## 📸 Features

| Feature | Description |
|---|---|
| 🎯 Risk Gauge | Visual risk score with colour-coded severity |
| 🕸️ Radar Chart | Per-factor risk breakdown |
| 💡 Smart Tips | Personalised intervention recommendations |
| 📊 Data Dashboard | Dataset exploration and correlation heatmap |
| 🏆 Model Comparison | MLR vs Decision Tree vs Random Forest |

---

## 🗂️ Project Structure

```
college-dropout-risk/
│
├── app.py                          ← Streamlit frontend (main app)
├── dropout_risk.pkl                ← Trained Random Forest pipeline
├── dropout_risk_cleaned.csv        ← Cleaned dataset (500 records)
│
├── notebooks/
│   ├── data_cleaning.ipynb         ← Data preprocessing
│   ├── mlr.ipynb                   ← Multiple Linear Regression
│   ├── decision.ipynb              ← Decision Tree Regressor
│   └── random.ipynb                ← Random Forest (best model)
│
├── requirements.txt
└── README.md
```

---

## 🔬 Dataset

- **Size**: 500 student records
- **Target**: `dropout_risk` — continuous value [0, 1]
- **Source**: Synthetic dataset with realistic distributions

### Features

| Feature | Type | Description |
|---|---|---|
| `current_gpa` | Float | GPA on a 0.0 – 4.0 scale |
| `attendance_rate` | Float | % of classes attended |
| `study_hours_per_week` | Float | Average weekly study hours |
| `financial_stress` | Int | Financial stress (1 = low, 10 = high) |
| `mental_health_score` | Int | Mental health (1 = poor, 10 = excellent) |
| `part_time_job` | Binary | 1 if student works part-time |
| `parental_support` | Categorical | low / medium / high |

---

## 🧠 ML Pipeline

```
Raw Data → Cleaning → Feature Encoding → Model Training → Prediction
```

### Models Compared

| Model | R² Score |
|---|---|
| Multiple Linear Regression | 0.427 |
| Decision Tree Regressor | 0.796 |
| **Random Forest Regressor** | **0.909** ✅ |

### Preprocessing
- Missing values → mean/median imputation
- `part_time_job` → binary encoding (Yes=1, No=0)
- `parental_support` → One-Hot Encoding via `ColumnTransformer`
- Full sklearn `Pipeline` (preprocessing + model in one object)

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/college-dropout-risk.git
cd college-dropout-risk

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit>=1.28
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
plotly>=5.18
```

---

## 💡 Key Learnings

1. **Random Forest** significantly outperforms linear models for this dataset — suggesting non-linear relationships between features and dropout risk
2. **Financial stress** and **mental health** are among the most important predictors
3. **Pipeline objects** make deployment clean — preprocessing and model are bundled together
4. **Early intervention** is the real-world goal: flagging students *before* they drop out

---

## 🔮 Future Improvements

- [ ] Add SHAP (SHapley values) for explainable AI per-student breakdowns
- [ ] Hyperparameter tuning with `GridSearchCV`
- [ ] Time-series dropout prediction (semester-by-semester)
- [ ] Integration with college ERP systems via REST API
- [ ] Batch prediction: upload a CSV of students and download risk scores

---

## 👤 Author

**[Your Name]**  
3rd Year Computer Science Student  
📧 your.email@example.com  
🔗 [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
