import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('all')

analyzer = SentimentIntensityAnalyzer()

st.title("Sentiment analysis")

sentence = st.text_input(
    "Write sentence here"
)

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button("Submit", on_click=click_button)

if st.session_state.clicked:
    scores = analyzer.polarity_scores(sentence)
    
    sentiment = "Neutral"

    if scores['compound'] > 0.5:
        sentiment = "Positive"
    elif scores['compound'] < -0.5:
        sentiment = "Negative"

    "Sentiment: " + sentiment

    if st.checkbox("Show scores"):
        "Negative: " + str(scores['neg'])
        "Neutral: " + str(scores['neu'])
        "Positive: " + str(scores['pos'])
        "Compound: " + str(scores['compound'])

    if st.checkbox("Show bar chart"):
        chart_data = pd.Series(scores)

        st.bar_chart(chart_data)