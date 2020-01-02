#@author:chenxinye
#@2019.06.09

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


df = pd.read_csv("text/df.csv")
#CNJ = pd.read_csv("Combined_News_DJIA.csv")
#Dt = pd.read_csv("DJIA_table.csv")

df.Label.value_counts()
x_train, x_test, y_train, y_test = train_test_split(df['combine_topic'],
                                                    df.Label,
                                                    test_size=0.5,
                                                    random_state=2019)

tfid_model = TfidfVectorizer(analyzer='word', 
                             stop_words='english',
                             max_features = 1000000,
                             ngram_range=(1,4)
                             ,min_df=0.01,
                             max_df=1.0)

x_train = tfid_model.fit_transform(x_train)
x_test = tfid_model.transform(x_test)

#print(x_train.todense().shape) 
#print(tfid_model.vocabulary_) 

mnb = MultinomialNB()
mnb.fit(x_train.todense(), y_train)
mnbpredict = mnb.predict(x_test.todense())
mnbpro = mnb.predict_proba(x_test.todense())[:,1]

print("beyes evaluation:\n", classification_report(y_test,mnbpredict))
print("beyes AUC:",roc_auc_score(y_test,mnbpro))

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C=30)
log.fit(x_train.todense(), y_train)
logpredict = log.predict(x_test.todense())
logpro = log.predict_proba(x_test.todense())[:,1]

print("logistic evaluation:\n", classification_report(y_test,logpredict))
print("logistic AUC:",roc_auc_score(y_test,logpro))

