"""
Created on Sun Apr 14 19:05:12 2013

@author1: Mohamed Aly <mohamed@mohamedaly.info>
@author2: Mahmoud Nabil <mah.nabil@yahoo.com>

"""



import cPickle as pickle
import numpy as np
from labr import *
import os
from qalsadi import analex
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn import metrics
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble.forest import RandomForestClassifier
from numpy.lib.scimath import sqrt
from numpy.ma.core import floor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcess
from sklearn import svm
from sklearn import preprocessing
from pickle import FALSE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import TruncatedSVD
LoadValidation = True  # Load The validation set
Evaluate_On_TestSet = False  # Evaluate either on evaluation or on test set if LoadValidation is True
Extract_Features = False  # Apply Feature Extraction techniques
Two_Stages_Classification=False
CrossValidation=False
UseLexicon=False
# data sets
datas = [
            dict(name="2-balanced", params=dict(klass="2", balanced="balanced")),
            dict(name="2-unbalanced",params=dict(klass="2",  balanced="unbalanced")),
            dict(name="3-balanced", params=dict(klass="3",   balanced="balanced")),
            dict(name="3-unbalanced", params=dict(klass="3", balanced="unbalanced")),
            dict(name="5-balanced", params=dict(klass="5", balanced="balanced")),
            dict(name="5-unbalanced",params=dict(klass="5", balanced="unbalanced"))
        ]

# tokenizer
an = analex.analex()
tokenizer = an.text_tokenize

# features
Features_Generators = [
                dict(name="count_ng1",
                feat_generator=CountVectorizer(tokenizer=tokenizer, ngram_range=(1,1))),
                dict(name="count_ng2",
                feat_generator=CountVectorizer(tokenizer=tokenizer, ngram_range=(1,2))),
                dict(name="count_ng3",
                feat_generator=CountVectorizer(tokenizer=tokenizer, ngram_range=(1,3))),
                dict(name="tfidf_ng1",
                feat_generator=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1, 1))),
                dict(name="tfidf_ng2",
                feat_generator=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,2))),
                dict(name="tfidf_ng3",
                feat_generator=TfidfVectorizer(tokenizer=tokenizer, ngram_range=(1,3))),

           ]

# classifiers
classifiers = [
#                 dict(name="svm_rbf",parameter_tunning=False, tune_clf=GridSearchCV(svm.SVC(tol=1e-3,kernel='rbf',cache_size=500),[{'C': [1,10,100] , 'gamma':[0.1,1,10]}] , cv=3 ) ,clf=svm.SVC(tol=1e-3,kernel='rbf',C=20,cache_size=500,gamma=0.8)), 
#                 dict(name="svm", parameter_tunning=False, clf=LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-3)),
                dict(name="Logistic Regression", parameter_tunning=False, tune_clf=GridSearchCV(LogisticRegression(), [{'penalty': ['l2'], 'C': [1, 10, 100]}], cv=3) , clf=LogisticRegression(penalty='l2', C=1)),
                dict(name="Passive Aggresive",parameter_tunning=False, clf = PassiveAggressiveClassifier(n_iter=100)),
                dict(name="SVM", parameter_tunning=False, clf=LinearSVC(loss='l2', penalty="l2", dual=False, tol=1e-3)),
                dict(name="Perceptron",parameter_tunning=False, clf = Perceptron(n_iter=100)),
#                
                dict(name="bnb",parameter_tunning=False,clf=BernoulliNB(binarize=0.5)),
                dict(name="sgd",parameter_tunning=False,clf=SGDClassifier(loss="hinge", penalty="l2")),
                dict(name="KNN",parameter_tunning=False,tune_clf=GridSearchCV( KNeighborsClassifier(),[{'n_neighbors': [5,10,50,100],'metric':['euclidean','minkowski'],'p':[2,3,4,5]}],cv=5 ) ,clf=KNeighborsClassifier(n_neighbors=3,metric='euclidean')),
#                 dict(name="KNN",parameter_tunning=False,tune_clf=GridSearchCV( KNeighborsClassifier(),[{'n_neighbors': [5,10,50,100],'metric':['euclidean','minkowski'],'p':[2,3,4,5]}],cv=5 ) ,clf=KNeighborsClassifier(n_neighbors=10,metric='euclidean')),
#                 dict(name="KNN",parameter_tunning=False,tune_clf=GridSearchCV( KNeighborsClassifier(),[{'n_neighbors': [5,10,50,100],'metric':['euclidean','minkowski'],'p':[2,3,4,5]}],cv=5 ) ,clf=KNeighborsClassifier(n_neighbors=100,metric='euclidean')),
#                 dict(name="KNN",parameter_tunning=False,tune_clf=GridSearchCV( KNeighborsClassifier(),[{'n_neighbors': [5,10,50,100],'metric':['euclidean','minkowski'],'p':[2,3,4,5]}],cv=5 ) ,clf=KNeighborsClassifier(n_neighbors=1000,metric='euclidean')),                
#                 dict(name="KNN",parameter_tunning=False,tune_clf=GridSearchCV( KNeighborsClassifier(),[{'n_neighbors': [5,10,50,100],'metric':['euclidean','minkowski'],'p':[2,3,4,5]}],cv=5 ) ,clf=KNeighborsClassifier(n_neighbors=5000,metric='euclidean')),
 
                dict(name="forest30",parameter_tunning=False,clf=RandomForestClassifier(n_estimators=30,random_state=123,verbose=3)),
                dict(name="lasso",parameter_tunning=False, clf=Lasso(alpha=1.0,tol=1e-3,warm_start = True)),
                dict(name="grad_boosting",parameter_tunning=False, clf=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=0)),
                dict(name="gaussian_process",parameter_tunning=False, clf=GaussianProcess(theta0=5e-1)),
                dict(name="ada_boost",parameter_tunning=False, clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)),
              ]

