# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:05:12 2013

@author1: Mohamed Aly <mohamed@mohamedaly.info>
@author2: Mahmoud Nabil <mah.nabil@yahoo.com>

"""



from Definations import *
from Utilities import *
from sklearn.feature_selection.univariate_selection import SelectPercentile
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.manifold import isomap
import numpy as np
from scipy.sparse import hstack


gr = LABR()
scores = list()


for data in datas:
###################################load the data####################################
        print(60 * "-")
        print("Loading data:", data['name'])
        if (LoadValidation):
            (d_train, y_train, d_test, y_test, d_valid , y_valid) = gr.get_train_test_validation(**data['params'])
            if (Evaluate_On_TestSet):
                d_train = np.concatenate((d_train, d_valid))
                y_train = np.concatenate((y_train, y_valid))
            else:
                d_test = d_valid
                y_test = y_valid
        else:
            (d_train, y_train, d_test, y_test) = gr.get_train_test(**data['params'])
####################################################################################       
     
        for feat_generator in Features_Generators:
####################################Features Generation#############################    
                print("Features Generation:", feat_generator['name'])
                X_train = feat_generator['feat_generator'].fit_transform(d_train)
                X_test = feat_generator['feat_generator'].transform(d_test)
                if(UseLexicon):
                    lexicon1=ReadLexicon()
                    lexicon_vectorizer1 = TfidfVectorizer(tokenizer=tokenizer,min_df=1,vocabulary=lexicon1,ngram_range=(1, 3),binary=True)
                    X_lexicon_train1 = lexicon_vectorizer1.fit_transform(d_train)
                    X_lexicon_test1 = lexicon_vectorizer1.transform(d_test)
                    lexicon2=ReadLexicon1()
                    lexicon_vectorizer2 = TfidfVectorizer(tokenizer=tokenizer,min_df=1,vocabulary=lexicon2,ngram_range=(1, 3),binary=True)
                    X_lexicon_train2 = lexicon_vectorizer2.fit_transform(d_train)
                    X_lexicon_test2 = lexicon_vectorizer2.transform(d_test)
                    X_train1=hstack((X_lexicon_train1, X_lexicon_train2))
                    X_test1=hstack((X_lexicon_test1, X_lexicon_test2))
#                     X_train1=hstack((X_train, X_train1))
#                     X_test1=hstack((X_test, X_test1))
                    X_train = X_train1
                    X_test =  X_test1
                    X_train=X_train.tocsr()
                    X_test=X_test.tocsr()                
                    
                for clf in classifiers:                    
                            pred = Train_And_Predict(X_train, y_train, X_test, clf['clf'], clf["name"])            
                            
                            
                            (acc, tacc, support, f1) = Evaluate_Result(pred, y_test)
                    
                            score = dict(data=data['name'],
                                            feat_generator=feat_generator['name'],
                                            clf=clf['name'],
                    #                              feat_ext=feat_ext['name'],
                                            f1=f1,
                                            acc=acc,
                                            tacc=tacc)    
            
                                
                            scores.append(score)
####################################Testing##############################################
print(60 * "=")
for s in scores:
    print("")
    for k, v in s.iteritems():
        print(k, v)





