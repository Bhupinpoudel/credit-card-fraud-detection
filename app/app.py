import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ==============================
# LOAD MODEL
# ==============================

model = pickle.load(open('fraud_model_app.pkl', 'rb'))

# ==============================
# TITLE
# ==============================

st.title("💳 Credit Card Fraud Detection System")
st.info("Machine Learning Model: Decision Tree with SMOTE")

# ==============================
# TABS
# ==============================

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Dashboard", "📁 Upload CSV"])

# ==============================
# TAB 1 → PREDICT
# ==============================

with tab1:

    # ==============================
    # CENTERED CONTAINER
    # ==============================

    left, center, right = st.columns([1, 2, 1])

    with center:

        st.markdown("### 🔍 Fraud Prediction")

        st.caption("Enter details and click predict")

        st.markdown("---")

        # ==============================
        # INPUTS (COMPACT)
        # ==============================

        col1, col2 = st.columns(2)

        with col1:
            time = st.number_input("Time", value=10000.0, label_visibility="visible")
            amount = st.number_input("Amount", value=100.0)

        with col2:
            v1 = st.number_input("V1", value=0.0)
            v2 = st.number_input("V2", value=0.0)
            v3 = st.number_input("V3", value=0.0)
            v4 = st.number_input("V4", value=0.0)
            v5 = st.number_input("V5", value=0.0)

        st.caption("V1–V5 are anonymised PCA features")

        st.markdown(" ")

        # ==============================
        # BUTTON (COMPACT)
        # ==============================

        predict_clicked = st.button("Predict", use_container_width=True)

        # ==============================
        # RESULT
        # ==============================

        if predict_clicked:

            input_data = np.array([[time, amount, v1, v2, v3, v4, v5]])
            prob = model.predict_proba(input_data)[0][1]

            st.write("Fraud Probability:", prob)

            if prob > 0.1:
                st.error(f"Fraud Detected (Prob: {prob:.3f})")
            else:
                st.success(f"Legitimate (Prob: {prob:.3f})")


# ==============================
# TAB 2 → DASHBOARD
# ==============================

with tab2:

    st.subheader("📊 Model Performance Dashboard")

    # ==============================
    # KPI CARDS (VERY PROFESSIONAL)
    # ==============================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "0.9616")
    col2.metric("Precision", "0.0384")
    col3.metric("Recall", "0.8878")
    col4.metric("F1 Score", "0.0736")

    st.markdown("---")

    # ==============================
    # PERFORMANCE TABLE
    # ==============================

    metrics = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Without SMOTE": [0.9995, 0.9048, 0.7755, 0.8352],
        "With SMOTE": [0.9616, 0.0384, 0.8878, 0.0736]
    }

    df_metrics = pd.DataFrame(metrics)

    st.subheader("📋 Performance Comparison")
    st.dataframe(df_metrics, use_container_width=True)

    # ==============================
    # CHARTS IN COLUMNS (CLEAN LOOK)
    # ==============================

    colA, colB = st.columns(2)

    # BAR CHART
    with colA:
        st.subheader("📊 Metrics Comparison")

        fig, ax = plt.subplots(figsize=(5,3))  # smaller size
        df_metrics.set_index("Metric").plot(kind="bar", ax=ax)
        plt.xticks(rotation=0)
        st.pyplot(fig)

    # ROC CURVE
    with colB:
        st.subheader("📈 ROC Curve")

        y_true = [0,0,0,1,1,1]
        y_scores = [0.1,0.2,0.3,0.7,0.8,0.9]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots(figsize=(5,3))
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0,1], [0,1], linestyle='--')
        ax2.legend()
        st.pyplot(fig2)

    st.markdown("---")

    # ==============================
    # CONFUSION MATRICES SIDE-BY-SIDE
    # ==============================

    colC, colD = st.columns(2)

    with colC:
        st.subheader("🔵 Without SMOTE")

        cm1 = [[56864, 0],
               [22, 76]]

        fig3, ax3 = plt.subplots(figsize=(4,3))
        sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax3)
        st.pyplot(fig3)

    with colD:
        st.subheader("🔴 With SMOTE")

        cm2 = [[54500, 2300],
               [11, 87]]

        fig4, ax4 = plt.subplots(figsize=(4,3))
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Reds", ax=ax4)
        st.pyplot(fig4)

# ==============================
# TAB 3 → CSV
# ==============================

with tab3:

    st.subheader("Upload CSV for Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of Data:")
        st.dataframe(df.head())

        try:
            features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5']
            X = df[features]

            df['Prediction'] = model.predict(X)

            st.write("Results:")
            st.dataframe(df)

            st.success("Prediction completed!")

            # ✅ NOW SAFE (df exists)
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"Error: {e}")

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.caption("Developed for MSc Dissertation | Fraud Detection System")
