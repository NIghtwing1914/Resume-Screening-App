import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("UpdatedResumeDataSet.csv")
df['Category'] = df['Category'].astype('category')

vals = df["Category"].cat.codes

vectorizer = TfidfVectorizer(stop_words="english")
vectorizer.fit(df["Resume"])
vectorized_text = vectorizer.transform(df["Resume"])

X_train,X_test,Y_train,Y_test=train_test_split(vectorized_text,vals,test_size=0.3,random_state=1)

codes_to_categories = dict(enumerate(df['Category'].cat.categories))

clf = KNeighborsClassifier()
clf.fit(X_train,Y_train)
preds = clf.predict(X_test)
print(accuracy_score(preds,Y_test))

def cleanText(text):
  processedText = re.sub(r'http\S+', '', text)
  processedText = re.sub(r'@\S+', '', processedText)
  processedText = re.sub(r'#\S+', '', processedText)
  return processedText

def main():
  st.title("Resume Screening App")
  uploaded_file = st.file_uploader('Upload Resume', type=["txt","pdf"])

  if uploaded_file is not None:
    try:
      resume_text = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
      resume_text = uploaded_file.read().decode('latin1')

    cleaned_resume = cleanText(resume_text)
    input_features = vectorizer.transform([cleaned_resume])
    prediction = clf.predict(input_features)
    st.write(prediction)

    category = codes_to_categories[prediction[0]]
    st.write(category)

if __name__ == "__main__":
    main()