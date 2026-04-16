# ==============================
# IMPORT LIBRARIES
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import pickle

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("creditcard.csv")

# ==============================
# SELECT FEATURES (SIMPLIFIED)
# ==============================

features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5']

X = df[features]
y = df['Class']

# ==============================
# SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# APPLY SMOTE
# ==============================

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ==============================
# TRAIN MODEL
# ==============================

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train_sm, y_train_sm)

# ==============================
# EVALUATE MODEL
# ==============================

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================

pickle.dump(model, open('fraud_model_app.pkl', 'wb'))

print("✅ Model saved as fraud_model_app.pkl")
