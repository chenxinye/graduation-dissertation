#@author:chenxinye
#@2019.06.09

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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


em = pd.read_csv("emotion_ev/emotion_evaluation.csv")
em.columns = ['index', 'negative', 'positive', 'count', 'sentiment']
em_col = ["negative","positive","count","sentiment"]
traincol = ['sexual_explicit', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

text = pd.read_csv(r"text/text_combine.csv")
text_em = pd.read_csv(r"text/text_test.csv")[traincol]
df_n = pd.read_csv(r"text/df_numeric.csv")
target = text[['Label']]

uls_train_col = ['date',
                 'open',
                 'high', 
                 'low',
                 'rate_of_return',
                 'rate_of_return_change',
                 'rate_of_return_change_shift',
                 'change']

numeric_col = ['close',
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

add_feature = ['sum',
               'min',
               'max',
               'mean',
               'std',
               'percentile5','percentile25','percentile75','percentile95']

ldalist = ['lda1','lda2','lda3','lda4','lda5','lda6']

#Feature Engineering
df = df_n.copy()
df['sum'] = df[numeric_col].sum(axis=1)  
df['min'] = df[numeric_col].min(axis=1)
df['max'] = df[numeric_col].max(axis=1)
df['mean'] = df[numeric_col].mean(axis=1)
df['std'] = df[numeric_col].std(axis=1)
df['med'] = df[numeric_col].median(axis=1)
for i in [5,25,75,95]:
    df['percentile'+str(i)] = df[numeric_col].apply(lambda x: np.percentile(x, i), axis=1)

df.fillna(0)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

df_n = df.copy()
for feature in (numeric_col+add_feature):
    df_n[feature] = df_n[feature].astype('float32')

#df_n = df_n[df_n.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
df_n = df_n.replace([np.inf, -np.inf], np.nan).fillna(method = 'ffill')
pca_x = pca.fit_transform(df_n[numeric_col+add_feature])

df['pca1'] = pca_x[:,0]
df['pca2'] = pca_x[:,1]

cntVector = CountVectorizer(stop_words='english')
cntTf = cntVector.fit_transform(text.combine_topic)
lda = LatentDirichletAllocation(n_topics=6,learning_offset=50.,random_state=0)
docres = lda.fit_transform(cntTf)

df['lda1'] = docres[:,0]
df['lda2'] = docres[:,1]
df['lda3'] = docres[:,2]
df['lda4'] = docres[:,3]
df['lda5'] = docres[:,4]
df['lda6'] = docres[:,5]

df = df.replace([np.inf, -np.inf], np.nan).fillna(method = 'ffill')
df_nm = pd.concat([df[numeric_col+add_feature+ldalist],text_em,em[em_col]],axis = 1)

#validation an feature importacne
oof = np.zeros(len(df))
feature_importance_df = pd.DataFrame()

folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=2019)

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
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
cols = (feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
plt.figure(figsize=(14,14))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('GDBT Features importance(averaged over folds)')
plt.tight_layout()
plt.show()

np.save("data/df_nm.npy",df_nm)
np.save("data/target.npy",target)
