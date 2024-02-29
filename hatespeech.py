import pandas as pd
import numpy as np
from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split
from sklearn. tree import DecisionTreeClassifier

import nltk
import re
nltk.download('stopwords')
from nltk.util import pr
stemmer = nltk. SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))

data = pd.read_csv("/content/twitter_data.csv")
#To preview the data
print(data.head())

data["labels"] = data["class"]. map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
data = data[["tweet", "labels"]]
print(data. head())

data = data[["tweet", "labels"]]
print(data. head())

def clean (text):
 text = str (text). lower()
 text = re. sub('[.?]', '', text)
 text = re. sub('https?://\S+|www.\S+', '', text)
 text = re. sub('<.?>+', '', text)
 text = re. sub('[%s]' % re. escape(string. punctuation), '', text)
 text = re. sub('\n', '', text)
 text = re. sub('\w\d\w', '', text)
 text = [word for word in text.split(' ') if word not in stopword]
 text=" ". join(text)
 text = [stemmer. stem(word) for word in text. split(' ')]
 text=" ". join(text)
 return text
data["tweet"] = data["tweet"]. apply(clean)
print(data. head())

x = np. array(data["tweet"])
y = np. array(data["labels"])
cv = CountVectorizer()
X = cv. fit_transform(x)
# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Model building
model = DecisionTreeClassifier()
#Training the model
model. fit(X_train,y_train)

#Testing the model
y_pred = model. predict (X_test)
y_pred

#Accuracy Score of our model
from sklearn. metrics import accuracy_score
print (accuracy_score (y_test,y_pred))

#Predicting the outcome
inp = "bitch get up off me"
inp = cv.transform([inp]).toarray()
print(model.predict(inp))

inp = "It is really awesome"
inp = cv. transform([inp]). toarray()
print(model. predict(inp))