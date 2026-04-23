# 💳 Credit Card Fraud Detection System

## 📌 Overview
This project implements a machine learning-based system to detect fraudulent credit card transactions. It uses a Decision Tree classifier along with SMOTE to handle class imbalance and improve fraud detection performance.

---

## 🚀 Features
-Fraud detection using Decision Tree classifier
-Handling imbalanced dataset using SMOTE
-Hyperparameter tuning using GridSearchCV
-Performance analysis across multiple tree depths
-Interactive Streamlit web application
-Tableau dashboard for visual analytics
-CSV upload for bulk prediction

---

## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Matplotlib & Seaborn

---

📊 Model Performance
Model	Precision	Recall	F1 Score
Without SMOTE	0.90	0.77	0.83
SMOTE (Depth=5)	0.04	0.89	0.07
GridSearch (Depth=10)	0.07	0.83	0.13
Depth=15	0.16	0.81	0.26

⚠️ Accuracy is not reliable due to class imbalance. Precision and recall are more meaningful metrics.

---

## 📂 Project Structure

credit-card-fraud-detection/
│
├── app/
│ ├── app.py
│ ├── fraud_model_app.pkl
│
├── notebook/
│ └── fraud_detection.ipynb
│
├── README.md
├── requirements.txt


---

## ▶️ How to Run the Project

### 1. Clone the repository

git clone https://github.com/Bhupinpoudel/credit-card-fraud-detection.git

cd credit-card-fraud-detection


---

### 2. Install dependencies

pip install -r requirements.txt


---

### 3. Run the Streamlit app

streamlit run app/app.py


---

## 📈 Results
The model demonstrates strong performance in detecting fraudulent transactions, particularly in terms of recall, which is critical for fraud detection systems.

---

## 🎓 Academic Context
This project was developed as part of an MSc dissertation focusing on machine learning-based fraud detection and data visualization techniques.

---

## 👨‍💻 Author
Bhupin Poudel
