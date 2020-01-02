#@author:chenxinye
#@2019.06.09

#for the whole dataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

em = pd.read_csv("emotion_ev/emotion_evaluation.csv")
em.columns = ['index', 'negative', 'positive', 'count', 'sentiment']
em_col = ["negative","positive","count","sentiment"]
traincol = ['sexual_explicit', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

text = pd.read_csv("data/Combined_News_DJIA.csv")
text_em = pd.read_csv(r"text/text_test.csv")[traincol]
df_n = pd.read_csv(r"text/df_numeric.csv")
target = text[['Label']]

uls_train_col = ['date','open', 'high', 'low','rate_of_return','rate_of_return_change', 'rate_of_return_change_shift','change']

index_ = ['close',
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
    'vr','vr_6_sma'
    ]

param1 = {
    'bagging_freq': 30,
    'bagging_fraction': 0.9,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.91,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 90,
    'min_sum_hessian_in_leaf': 15.0,
    'num_leaves': 27,
    'num_threads': 20,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': -1
}

folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=859)

#Feature Engineering 1
#df_nm = pd.concat([df_n[index_]],axis = 1)
df_nm = pd.concat([df_n[index_],em[em_col],text_em,text_em[traincol]],axis = 1)
oof = np.zeros(len(df_nm))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_nm.values, target.values)):
    
    X_train, y_train = df_nm.iloc[trn_idx], target.iloc[trn_idx]
    X_valid, y_valid = df_nm.iloc[val_idx], target.iloc[val_idx]
    
    X_tr, y_tr = X_train, y_train
    X_tr = pd.DataFrame(X_tr)
    
    #print("Fold idx:{}".format(fold_ + 1))
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    
    clf = lgb.train(param1, trn_data, 2000, valid_sets = [val_data], verbose_eval=100000, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(df_nm .iloc[val_idx],num_iteration=clf.best_iteration)
    #print("CV score: {:<8.5f}".format(roc_auc_score(target.loc[val_idx], oof[val_idx])))
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = df_nm .columns
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
cols = (feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
plt.figure(figsize=(10,10))
featurescore = best_features.groupby("feature").mean().sort_values(by="importance",ascending=False).reset_index()
sns.barplot(x="importance", y="feature", data=featurescore)
plt.title('GDBT Features (averaged over folds)')
plt.tight_layout()
plt.show()

#Feature Engineering 2
#df_text = text['Top1'] 
#oof = np.zeros(len(df_nm))
#feature_importance_df = pd.DataFrame()
#
#vectorizer = CountVectorizer(max_features = 5000) 
#
#df_text = vectorizer.fit_transform(df_text)
#df_text = pd.DataFrame(df_text.todense())
#
#folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=859)
#for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_text.values, target.values)):
#    
#    X_train, y_train = df_text.iloc[trn_idx], target.iloc[trn_idx]
#    X_valid, y_valid = df_text.iloc[val_idx], target.iloc[val_idx]
#    
#    mnb = MultinomialNB()
#    mnb.fit(X_train, y_train)
#    print("Fold idx:{}".format(fold_ + 1))
#    oof[val_idx] = mnb.predict_proba(df_text.iloc[val_idx])[:,1]
#    print("CV score: {:<8.5f}".format(roc_auc_score(target.loc[val_idx], oof[val_idx])))
#    
#
##Feature Engineering 3
#df_text = text['Top1'] 
#oof = np.zeros(len(df_nm))
#feature_importance_df = pd.DataFrame()
#
#tfid_model = TfidfVectorizer(
#        analyzer='word', stop_words='english',max_features = 1000000,ngram_range=(1,4),min_df=0.01,max_df=1.0
#                             )
#
#df_text = tfid_model.fit_transform(df_text)
#df_text = pd.DataFrame(df_text.todense())
#
#folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=859)
#for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_text.values, target.values)):
#    
#    X_train, y_train = df_text.iloc[trn_idx], target.iloc[trn_idx]
#    X_valid, y_valid = df_text.iloc[val_idx], target.iloc[val_idx]
#    
#    mnb = MultinomialNB()
#    mnb.fit(X_train, y_train)
#    print("Fold idx:{}".format(fold_ + 1))
#    oof[val_idx] = mnb.predict_proba(df_text.iloc[val_idx])[:,1]
#    print("CV score: {:<8.5f}".format(roc_auc_score(target.loc[val_idx], oof[val_idx])))

#
#runfile('D:/毕业设计/stocknews/Feature_importance.py', wdir='D:/毕业设计/stocknews')
#Fold idx:1
#Training until validation scores don't improve for 500 rounds.
#Did not meet early stopping. Best iteration is:
#[1999]  valid_0's auc: 0.933816
#CV score: 0.93382 
#Fold idx:2
#Training until validation scores don't improve for 500 rounds.
#Did not meet early stopping. Best iteration is:
#[1998]  valid_0's auc: 0.920873
#CV score: 0.92087 
#Fold idx:3
#Training until validation scores don't improve for 500 rounds.
#Did not meet early stopping. Best iteration is:
#[1973]  valid_0's auc: 0.922421
#CV score: 0.92242 
#Fold idx:4
#Training until validation scores don't improve for 500 rounds.
#Did not meet early stopping. Best iteration is:
#[1739]  valid_0's auc: 0.950006
#CV score: 0.95001 
#Fold idx:5
#Training until validation scores don't improve for 500 rounds.
#Did not meet early stopping. Best iteration is:
#[1962]  valid_0's auc: 0.916871
#CV score: 0.91687 
#CV score: 0.92695 