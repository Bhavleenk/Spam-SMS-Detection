import streamlit as st
import pickle
import nltk
import string
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
# nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords


def TextTransform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Message Spam Classifier")
InputMessage = st.text_area("Enter the text here")
if st.button('Predict'):
    TransformMessage = TextTransform(InputMessage)
    VectorInput = tfidf.transform([TransformMessage])
    Result = model.predict(VectorInput)[0]
    if Result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
