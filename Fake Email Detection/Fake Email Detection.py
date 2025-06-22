import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv('spam.csv',encoding='latin')

df.head()

df.v1.unique()

df['spam']=df['v1'].apply(lambda x: 1 if x=='spam' else 0)

df.head(5)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.v2,df.spam,test_size=0.2,random_state=42)

len(x_train)

len(x_test)

from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
cv_messages = v.fit_transform(x_train.values)
cv_messages.toarray()[0:5]

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()

model.fit(cv_messages,y_train)

email = [
         'Upto 30% discount on parking, exclusive offer just for yoy. Dont miss thi reward!',
         'Ok lar...joking wif u oni...'
]
email_count= v.transform(email)
model.predict(email_count)

x_test_count=v.transform(x_test)
model.score(x_test_count,y_test)

"""# sklearn pipeline"""

from sklearn.pipeline import Pipeline
clf = Pipeline([
      ('vectorizer', CountVectorizer()),
      ('nb', MultinomialNB())
]
)
clf.fit(x_train,y_train)

email = [
        'Upto 30% discount on parking, exclusive offer just for yoy. Dont miss thi reward!',
         'Ok lar...joking wif u oni...'
]
clf.predict(email)

clf.score(x_test,y_test)

import joblib
joblib.dump(clf,'spam_model.pkl')

# model is completed
