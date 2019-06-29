#@author:chenxinye
#@2019.06.09

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

index_ = ['open', 'high', 'low', 'volume',
    'date','rate_of_return', 'rate_of_return_change','rate_of_return_change_shift',
    'close',
    'cr','cr-ma1','cr-ma2','cr-ma3',
    'kdjk','kdjd','kdjj',
    'close_5_sma','close_10_sma',
    'macd','macds','macdh',
    'boll','boll_ub','boll_lb',
    'rsi_6','rsi_12',
    'wr_10','wr_6',
    'cci','cci_20',
    'tr','atr',
    'dma',
    'trix','trix_9_sma',
    'vr','vr_6_sma','change']

df_numeric = pd.read_csv("text/df_numeric.csv")[index_]
df_text = pd.read_csv("text/df.csv")
#df_text.Label.head(10)
#df_numeric.change.head(10)
drop_index = ['date','rate_of_return', 'rate_of_return_change','rate_of_return_change_shift']
del df_numeric['change']

#check if it is ok
#index_f = ['close',
#    'cr','cr-ma1','cr-ma2','cr-ma3',
#    'kdjk','kdjd','kdjj',
#    'close_5_sma','close_10_sma',
#    'macd','macds','macdh',
#    'boll','boll_ub','boll_lb',
#    'rsi_6','rsi_12',
#    'wr_10','wr_6',
#    'cci','cci_20',
#    'tr','atr',
#    'dma',
#    'trix','trix_9_sma',
#    'vr','vr_6_sma'
#    ]

df_numeric = df_numeric.drop(drop_index,axis = 1)
df_numeric.columns
#df = pd.concat([df_text,df_numeric], axis=1)

tfid_model = TfidfVectorizer(analyzer='word', 
                             stop_words='english',
                             max_features = 1000000,
                             ngram_range=(1,4),
                             min_df=0.01,
                             max_df=1.0)

text_tfidf = pd.DataFrame(tfid_model.fit_transform(df_text.combine_topic).todense())
df_train = pd.concat([text_tfidf,df_numeric],axis = 1).fillna(-999)
df_train = df_train[np.isfinite(df_train)]
df_train = df_train.T.drop_duplicates(inplace=False).T

x_train, x_test, y_train, y_test = train_test_split(df_train.fillna(999),
                                                    df_text.Label,
                                                    test_size=0.5,
                                                    random_state=2019)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C=30)
log.fit(x_train, y_train)
logpredict = log.predict(x_test)
logpro = log.predict_proba(x_test)[:,1]

print("logistic evaluation:\n", classification_report(y_test,logpredict))
print("logistic AUC:",roc_auc_score(y_test,logpro))

from sklearn import svm
svm = svm.SVC(kernel='linear',probability = True)
svm.fit(X=x_train, y=y_train,sample_weight=None)
svmpredict = svm.predict(x_test)
svmpro = svm.predict_proba(x_test)[:,1]

print("svm evaluation:\n", classification_report(y_test,svmpredict))
print("svm AUC:",roc_auc_score(y_test,svmpro))

from sklearn.naive_bayes import GaussianNB
gau = GaussianNB()
gau.fit(x_train, y_train)
gaupredict = gau.predict(x_test)
gaupro = gau.predict_proba(x_test)[:,1]
print("svm evaluation:\n", classification_report(y_test,gaupredict))
print("svm AUC:",roc_auc_score(y_test,gaupro))
#'linear':线性核函数
#'poly'：多项式核函数
#'rbf'：径像核函数/高斯核
#'sigmod':sigmod核函数
#'precomputed':核矩阵
#print('支持向量：',svm.support_vectors_)
## 获得支持向量的索引
#print('支持向量索引：',svm.support_)
## 为每一个类别获得支持向量的数量
#print('支持向量数量：',svm.n_support_)

#since the auc score is so high that i suspect that the data contain leak, so i need to find it by correlation metrix
#correlations = df_numeric.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
#correlations = correlations[correlations['level_0'] != correlations['level_1']]
#correlations.columns = ['index1','index2','correlation']
#corr = correlations.sort_values(by = ['correlation'],axis = 0, ascending=False)
#corr[corr.index1 == 'change']

