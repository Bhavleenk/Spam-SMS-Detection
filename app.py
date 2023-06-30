import streamlit as st
import pickle
import nltk
import string
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
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
st.markdown('---')
InputMessage = st.text_area("Enter the text here")
if st.button('Predict'):
    TransformMessage = TextTransform(InputMessage)
    VectorInput = tfidf.transform([TransformMessage])
    Result = model.predict(VectorInput)[0]
    if Result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

st.markdown('---')
st.markdown("### **Example SPAM:**")
st.markdown(' - WINNER!! As a valued customer you have been selected to received 900 credits!')
st.markdown(" - Last chances to win CASH REWARD! ")
st.markdown("### **Example NOT SPAM:**")
st.markdown(" - Your bill for this month is $400")
st.markdown(" - I HAVE A DATE THIS SUNDAY!!")
st.markdown('---')
st.markdown("*Made by **Bhavleen Kaur***")
st.markdown("*Github: **github.com/Bhavleenk***")
st.markdown('---')

