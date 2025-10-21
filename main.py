import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')

# Load trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# Function to preprocess input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()
    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]
    y.clear()
    y = [ps.stem(i) for i in text]
    return ' '.join(y)

# Streamlit Sidebar Navigation
st.set_page_config(page_title='Spam Mail Detection', page_icon='📧', layout='wide')
pages = ['Home', 'Spam Mail Detection', 'Evaluation Metrics', 'About']
selected_page = st.sidebar.selectbox('Navigation', pages)

# ---------------------- Home Page ----------------------
if selected_page == 'Home':
    st.title('📧 Spam Mail Detection App')
    st.write("This app helps detect whether an email is **Spam** or **Ham**.")
    st.write("Use the sidebar to navigate between pages: Spam Detection, Evaluation Metrics, and About.")

# ---------------------- Spam Mail Detection Page ----------------------
elif selected_page == 'Spam Mail Detection':
    st.title('📨 Spam Mail Detection')
    st.write('Enter the email text below or upload a .txt file to check if it is Spam or Ham.')

    # Option 1: Text input
    user_input = st.text_area('Email Content')

    # Option 2: File upload
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
    file_content = None
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")

    # Use either text area input or uploaded file content
    input_text = file_content if file_content else user_input

    if st.button('Predict'):
        if not input_text or input_text.strip() == '':
            st.warning('Please enter some email text or upload a .txt file.')
        else:
            transformed_input = transform_text(input_text)
            vectorized_input = tfidf.transform([transformed_input]).toarray()
            prediction = model.predict(vectorized_input)[0]
            prediction_prob = model.predict_proba(vectorized_input)[0][prediction] if hasattr(model, 'predict_proba') else None

            if prediction == 1:
                st.error(f'🚨 Spam Email! Probability: {prediction_prob:.2f}' if prediction_prob else '🚨 Spam Email!')
            else:
                st.success(f'✅ Ham Email! Probability: {prediction_prob:.2f}' if prediction_prob else '✅ Ham Email!')

# ---------------------- Evaluation Metrics Page ----------------------
elif selected_page == 'Evaluation Metrics':
    st.title('📊 Model Evaluation Metrics')
    st.write('View performance metrics of the trained spam detection model.')

    # Load dataset for evaluation
    df = pd.read_csv('spam.csv', encoding='latin1')
    df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
    df.rename(columns={'v1':'target','v2':'text'}, inplace=True)
    df.drop_duplicates(inplace=True)
    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])
    df['transformed_text'] = df['text'].apply(transform_text)
    X_eval = tfidf.transform(df['transformed_text']).toarray()
    y_eval = df['target'].values
    y_pred_eval = model.predict(X_eval)

    st.subheader('Accuracy')
    st.write(accuracy_score(y_eval, y_pred_eval))

    st.subheader('Precision')
    st.write(precision_score(y_eval, y_pred_eval))

    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_eval, y_pred_eval)
    st.write(cm)

    st.subheader('Class Distribution')
    st.bar_chart(df['target'].value_counts())

# ---------------------- About Page ----------------------
elif selected_page == 'About':
    st.title('ℹ️ About')
    st.write('This Spam Mail Detection app is built using:')
    st.write("""
- Python
- Streamlit
- scikit-learn
- NLTK
- TF-IDF Vectorization
- Multinomial Naive Bayes Model
""")
    st.write('The app provides a user-friendly interface to check emails for spam and displays evaluation metrics.')