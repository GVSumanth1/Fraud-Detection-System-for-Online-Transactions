# Fraud Detection System for Online Transactions


## Why I thought of this project
- Fraud is **rare and evolving** – traditional rule-based systems cannot keep up.  
- Machine learning can **learn fraud patterns automatically** from data.  
- My goal was to build a **robust fraud detection model** that balances recall and precision.  

---

## Dataset I used
- **Source**: IEEE-CIS Fraud Detection dataset (Kaggle, provided by Vesta Corporation)  
- **Size**: 590,540 transactions  
- **Features**: 434 columns (transaction, identity, categorical, anonymized, time, address/geo info)  
- **Target Variable**: `isFraud` (1 = Fraudulent, 0 = Legitimate)  
- **Class Imbalance**: Fraud cases only ~3.5% of total data  

---

## What I did

### Data Preprocessing
- Handled missing values  
- Encoded categorical variables  
- Transformed columns where required  
- Checked class imbalance carefully  

### Exploratory Analysis
- Visualized fraud vs non-fraud transaction counts  
- Analyzed fraud patterns by **day, hour, and amount**  
- Found riskier devices, email domains, and OS/browser combinations  
- Observed that **transaction amount and time** are strong indicators  

### Model Training
I experimented with different models:  
- **Logistic Regression** – simple baseline  
- **KNN** – detected local anomalies but slow on large data  
- **Random Forest** – good performance but heavy computation  
- **LightGBM** – very fast and accurate  
- **XGBoost** – best performance overall  

---

## Results

### Random Forest
- ROC-AUC: **0.877**  
- Accuracy: **97%**  
- Fraud recall was low (only 30%), so many frauds were missed  

### LightGBM
- Validation AUC: **0.9656**  
- Recall: **81%** (caught most frauds)  
- Precision: **53%** (some false alarms)  
- F1 Score: **64%**  

### XGBoost (Best Model)
- Validation AUC: **0.972**  
- Accuracy: **99%**  
- Fraud Precision: **0.89** (very few false alarms)  
- Fraud Recall: **0.78** (caught majority of frauds)  
- F1 Score: **0.83** – strong balance  
- Final choice for production  

---

## Predictor App
- Built a simple **Streamlit app** for fraud prediction  
- Based on XGBoost with reduced feature set  
- Run the app:  
  ```bash
  streamlit run app.py

## What I learned
-Handling imbalanced datasets is critical in fraud detection
-Tree-based models like XGBoost and LightGBM are best for complex tabular data
-Feature importance helps in understanding fraud indicators
-Building a small app interface makes the project more practical
