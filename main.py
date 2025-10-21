import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from nltk.tokenize import RegexpTokenizer

# -------------------- Setup --------------------
# Initialize stemmer and tokenizer
ps = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

# Download stopwords
nltk.download('stopwords', quiet=True)

# Load trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -------------------- Functions --------------------
def transform_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)  # regex tokenizer
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [ps.stem(t) for t in tokens]
    return ' '.join(tokens)

# -------------------- Streamlit App --------------------
st.set_page_config(page_title='Spam Mail Detection', page_icon='📧', layout='wide')
pages = ['Home', 'Spam Mail Detection', 'Evaluation Metrics', 'About']
selected_page = st.sidebar.selectbox('Navigation', pages)

# -------------------- Home Page --------------------
if selected_page == 'Home':
    st.title('📧 Spam Mail Detection App')
    st.write("This app detects whether an email is **Spam** or **Ham**.")
    st.write("Use the sidebar to navigate between pages: Spam Detection, Evaluation Metrics, and About.")

# -------------------- Spam Mail Detection --------------------
elif selected_page == 'Spam Mail Detection':
    st.title('📨 Spam Mail Detection')
    st.write('Enter the email text below or upload a .txt file to check if it is Spam or Ham.')

    # Text input
    user_input = st.text_area('Email Content')

    # File upload
    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
    file_content = None
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read().decode("utf-8")
        except:
            st.error("Unable to read the uploaded file. Make sure it is a valid .txt file.")

    # Use uploaded file if available, else use text area input
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

# -------------------- Evaluation Metrics --------------------
elif selected_page == 'Evaluation Metrics':
    st.title('📊 Model Evaluation Metrics')
    st.write('Performance metrics of the trained spam detection model.')

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

# -------------------- About --------------------
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
    st.write('It allows text input or .txt file upload for spam detection and shows evaluation metrics.')
