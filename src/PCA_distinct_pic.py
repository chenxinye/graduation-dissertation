#@author:chenxinye
#@2019.06.09

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

SEED = 2019

em = pd.read_csv("emotion_ev/emotion_evaluation.csv")
em.columns = ['index', 'negative', 'positive', 'count', 'sentiment']
em_col = ["negative","positive","sentiment"]

text = pd.read_csv("data/Combined_News_DJIA.csv")
df_n = pd.read_csv("text/df_numeric.csv")
target = text[['Label']]
traincol = ['sexual_explicit', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

uls_train_col = ['date','open', 'high', 'low','rate_of_return','rate_of_return_change', 'rate_of_return_change_shift','change']
text_em = pd.read_csv(r"text\text_test.csv")[traincol]
pca_test = pd.concat([em[em_col],text_em],axis = 1)
pca_test_all = pd.concat([em[em_col],text_em,target],axis = 1)

#pca_test = (pca_test - pca_test.mean())/pca_test.std()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
test = pca.fit_transform(pca_test)
data = pd.concat([pd.DataFrame(test),target],axis = 1)
data.columns = ['x','y','Label']

import seaborn as sns  
import matplotlib.pyplot as plt
#plt.style.use({'figure.figsize':(20, 20)})
sns.pairplot(data, vars=["x", "y"],hue='Label',palette="husl",height = 5,kind = 'scatter')
plt.figure(figsize = (20,20))
plt.show() 