#@author:chenxinye
#@2019.06.09

# A host of Scikit-learn models
#severe_toxicity
#obscene
#threat
#insult
#identity_attack
#sexual_explicit
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
import pydotplus  # you can install pydotplus with: pip install pydotplus
from IPython.display import Image
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn import svm
import time 
import warnings
warnings.filterwarnings('ignore')

SEED = 2019
df_nm = np.load("data/df_nm.npy")
target = np.load("data/target.npy")

xtrain, xtest, ytrain, ytest = train_test_split(df_nm,target,test_size=0.5,random_state=SEED)

def get_models():
    """Generate a library of base learners."""
    svc = svm.SVC(kernel='linear',probability = True)
    nn = MLPClassifier((40, 20), early_stopping=False, random_state=SEED)
    dt = DecisionTreeClassifier(max_depth=35, random_state=SEED)
    lr = LogisticRegression(C=90, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, max_features=30, random_state=SEED)
    xgm = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=0.65, gamma=2, learning_rate=0.01, max_delta_step=1,
               max_depth=30, min_child_weight=2, missing=None, n_estimators=200,
               n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=SEED,
               silent=True, subsample=1)
    
    models = {'SVM':svc,
              'Neural Network':nn,
              'decisiontree':dt,
              'random forest': rf,
              'xgboost': xgm,
              'logistic': lr,
              }

    return models


def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((ytest.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(xtrain, ytrain)
        P.iloc[:, i] = m.predict_proba(xtest)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

models = get_models()
P = train_predict(models)
score_models(P, ytest)

# You need ML-Ensemble for this figure: you can install it with: pip install mlens
from mlens.visualization import corrmat

corrmat(P.corr(), inflate=False)
plt.show()


print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(ytest, P.mean(axis=1)))

from sklearn.metrics import roc_curve

def plot_roc_curve(ytest, P_base_learners, P_ensemble, labels, ens_label):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    cm = [plt.cm.rainbow(i)
      for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]

    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(ytest, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    fpr, tpr, _ = roc_curve(ytest, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.show()

class model_stacking:
    def __init__(self,train,target):
        self.train = train
        self.target = target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(train,target,test_size=0.5,random_state=SEED)
        self.tr_pred,self.testpred,self.y_train = self.train_model()
        self.pred = self.stacking_train()
        
    def model_blend(self):
        svc = svm.SVC(kernel='linear',probability = True)
        nn = MLPClassifier((40, 20), early_stopping=False, random_state=SEED)
        dt = DecisionTreeClassifier(max_depth=35, random_state=SEED)
        lr = LogisticRegression(C=90, random_state=SEED)
        rf = RandomForestClassifier(n_estimators=300, max_features=30, random_state=SEED)
        xgm = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                   colsample_bytree=0.65, gamma=2, learning_rate=0.01, max_delta_step=1,
                   max_depth=30, min_child_weight=2, missing=None, n_estimators=200,
                   n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                   reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=SEED,
                   silent=True, subsample=1)
        
        model_list = [svc,nn,dt,lr,rf,xgm]
        
        return model_list
    
    def train_model(self):
        model_list = self.model_blend()
        print("begin")
        x_train,y_train = self.x_train,self.y_train
        for model in model_list:
            print(model)
            if model == model_list[0]:
                model.fit(x_train,y_train)
                pred = model.predict_proba(x_train)[:,1].reshape(-1,1)
                testpred = model.predict_proba(self.x_test)[:,1].reshape(-1,1)
            else:
                model.fit(x_train,y_train)
                pre = model.predict_proba(x_train)[:,1]
                pred = np.hstack((pred,pre.reshape(-1,1)))
                testpre = model.predict_proba(self.x_test)[:,1]
                testpred = np.hstack((testpred,testpre.reshape(-1,1)))
        print("done!")
        
        return (pred,testpred,y_train)

    def stacking_train(self):
        mnb = MultinomialNB()
        #log = LogisticRegression(C=100, random_state=42)
        mnb.fit(self.tr_pred,self.y_train)
        #log.fit(tr_pred,y_train)
        test_pred = mnb.predict_proba(self.testpred)[:,1]
        print("stacking AUC:",roc_auc_score(self.y_test,test_pred))
        return test_pred

def log_meta(self):
    log = LogisticRegression(C=100, random_state=42)
    log.fit(self.tr_pred,self.y_train)
    test_pred = log.predict_proba(self.testpred)[:,1]
    print("stacking AUC:",roc_auc_score(self.y_test,test_pred))
    return test_pred

def xgb_meta(self):
    xgm = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                   colsample_bytree=0.65, gamma=2, learning_rate=0.01, max_delta_step=1,
                   max_depth=4, min_child_weight=2, missing=None, n_estimators=200,
                   n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                   reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=SEED,
                   silent=True, subsample=1)
    xgm.fit(self.tr_pred,self.y_train)
    test_pred = xgm.predict_proba(self.testpred)[:,1]
    print("stacking AUC:",roc_auc_score(self.y_test,test_pred))
    return test_pred

pipeline_stacking = model_stacking(df_nm,target)
print("gau model:")
p = pipeline_stacking.stacking_train()
print("logistic model:")
p1 = log_meta(pipeline_stacking)
print("xgb model:")
p2 = xgb_meta(pipeline_stacking)
plot_roc_curve(ytest, P.values, P.mean(axis=1), list(P.columns), "ensemble")

#==================
#产生信号
#==================
x_train, x_test, y_train, y_test = df_nm[0:int(0.5*len(df_nm))],df_nm[int(0.5*len(df_nm)):],target[0:int(0.5*len(df_nm))],target[int(0.5*len(df_nm)):]


def final_models():
    """Generate a library of base learners."""
    svc = svm.SVC(kernel='linear',probability = True)
    lr = LogisticRegression(C=90, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, max_features=30, random_state=SEED)
    xgm = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
               colsample_bytree=0.65, gamma=2, learning_rate=0.01, max_delta_step=1,
               max_depth=25, min_child_weight=2, missing=None, n_estimators=500,
               n_jobs=1, nthread=None, objective='binary:logistic', random_state=SEED,
               reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=SEED,
               silent=True, subsample=1)
    
    models = {'SVM':svc,
              'random forest': rf,
              'xgboost': xgm,
              'logistic': lr,
              }

    return models

def final_train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((y_test.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(model_list.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(x_train, y_train)
        P.iloc[:, i] = m.predict_proba(x_test)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

start = time.time()
final_model = final_models()
fP = final_train_predict(final_model)
score_models(fP, y_test)
end = time.time()
print("Ensemble ROC-AUC score: %.3f" % roc_auc_score(y_test, fP.mean(axis=1)))

print('time consumption:',end - start)

df_final_result = pd.read_csv("text\df_numeric.csv")[int(0.5*len(df_nm)):].reset_index()

df_final_result['finalsig'] = fP.mean(axis=1)

print("evaluation:\n", classification_report(y_test,np.round(fP.mean(axis=1))))

df_final_result.to_csv("src\output\signal.csv",index = 0)

from sklearn.metrics import confusion_matrix
confusion_matrix(np.array(y_test),np.round(fP.mean(axis=1)))
from sklearn.metrics import accuracy_score
accuracy_score(np.array(y_test),np.round(fP.mean(axis=1)))

def search_best(yt,yp):
    acc = accuracy_score(yt,np.round(yp))
    bst = 0.5
    for i in np.arange(0,1,0.01):
        y = yp.copy()
        y[y>i] = 1
        y[y<=i] = 0
        acc_ = accuracy_score(yt,y)
        print(acc_)
        if acc_> acc:
            bst = i
    yp[yp>bst] = 1
    yp[yp<bst] = 0
    return bst,yp

bst,y = search_best(y_test,fP.mean(axis=1))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(np.array(y_test),y))
from sklearn.metrics import accuracy_score
print(accuracy_score(np.array(y_test),y))