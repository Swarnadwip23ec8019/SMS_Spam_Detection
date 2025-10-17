import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Streamlit Page Config
st.set_page_config(page_title="Spam Mail Detection", page_icon="📧", layout="wide")

st.title("📨 Spam Mail Detection App (Machine Learning)")
st.write("This app trains a Logistic Regression model to classify emails as **Spam** or **Ham (Not Spam)**.")

# Sidebar
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file (must contain 'Category' and 'Message' columns):", type=['csv'])

# Load default dataset if none uploaded
if uploaded_file is not None:
    mail_data = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
else:
    st.info("Using default dataset: `mail_data.csv`")
    mail_data = pd.read_csv('mail_data.csv', encoding='ISO-8859-1')

# Replace null values
mail_data = mail_data.where((pd.notnull(mail_data)), '')

# Label encoding
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

X = mail_data['Message']
Y = mail_data['Category'].astype('int')

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# TF-IDF Feature Extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Evaluate Model
Y_test_pred = model.predict(X_test_features)

accuracy = accuracy_score(Y_test, Y_test_pred)
precision = precision_score(Y_test, Y_test_pred)
recall = recall_score(Y_test, Y_test_pred)
f1 = f1_score(Y_test, Y_test_pred)

# Display Metrics
st.subheader("📊 Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Precision", f"{precision*100:.2f}%")
col3.metric("Recall", f"{recall*100:.2f}%")
col4.metric("F1 Score", f"{f1*100:.2f}%")

# Confusion Matrix
st.subheader("📉 Confusion Matrix")
cm = confusion_matrix(Y_test, Y_test_pred)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spam','Ham'], yticklabels=['Spam','Ham'], ax=ax)
st.pyplot(fig)

# Classification Report
st.subheader("📋 Classification Report")
st.text(classification_report(Y_test, Y_test_pred, target_names=['Spam', 'Ham']))

# Spam Prediction Section
st.subheader("✉️ Try Out the Model")
input_mail = st.text_area("Enter an email/message text below:")

if st.button("Predict"):
    if input_mail.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # Transform input text
        input_data_features = feature_extraction.transform([input_mail])
        prediction = model.predict(input_data_features)[0]

        if prediction == 1:
            st.success("✅ This is a **HAM** message (Not Spam).")
        else:
            st.error("🚫 This is a **SPAM** message!")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Scikit-learn.")
