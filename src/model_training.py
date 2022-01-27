import re
import string
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import wandb
from wandb.keras import WandbCallback

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot, hashing_trick, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Activation, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout
from keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from util import load_data, preprocessing, confusion_matrix_plot, evaluate, get_vocab_size, create_token2id, one_hot_text, get_verdict, get_verdict_with_token2id
"""
Function for training and preprocessing our data on models
currently preprocessing based on bag of words
currently training models on Naive Bayes,logistic regression, decision tree, random forest, SVM, and stacking
Outputs trained models to the Models folder, given a name based on what category was given (general, politics, science)
"""
def prep_train_models(model = "all", cat = "general"):
    df = pd.read_csv("data/cleaned data - Sheet1.csv")
    #standardize category format and df category format
    cat = cat.lower()
    df["Category"] =df["Category"].str.lower()
    #checks for the category that we want models trained on
    if cat != "general":
        df = df[df["Category" ]== cat]
        #gets text and verdict
    simple_df = df[["Original article text", "Verdict"]]
    X = simple_df["Original article text"].values
    y = simple_df["Verdict"].replace("FALSE",1).replace("TRUE",0).values
    #preproccess into bag of words and save the pckl of the bag of words into models
    
    X_preproc = [preprocessing(i) for i in X]
    cv = CountVectorizer()
    X_cv = cv.fit_transform(X)
    with open('models/BoW.pickle', 'wb') as handle:
        pickle.dump(cv, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #naive bayes
    if cat == "general" or cat == "Naive Bayes":
        # Initializing the model
        naive_bayes = MultinomialNB(alpha=0.3, fit_prior = True)
        # Fitting the data into the model
        naive_bayes.fit(X_cv, y)
        pkl_filename = "models/naive_bayes_model" +"_" +cat+ ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(naive_bayes, file)
    #logisitic regression        
    if cat == "general" or cat == "logisitic regression":
        # Initializing the model
        logistic_regression = LogisticRegression(C=1, random_state = 0,  max_iter=1000)
        # Fitting the data into the model
        logistic_regression=logistic_regression.fit(X_cv, y)
        pkl_filename = "models/logistic_regression_model" + "_" +cat + ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(logistic_regression, file)
    #SVM
    if cat == "general" or cat == "svm":
        # Initializing the model
        svm = SVC(kernel = 'linear', C=1, random_state = 0)
        # Fitting the data into the model
        svm.fit(X_cv, y)
        pkl_filename = "models/svm_model" +"_" +cat+".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(svm, file)
     #Decision Tree
    if cat == "general" or cat == "decision tree":
        # Initializing the model
        decision_Tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 90, max_features = "sqrt", min_samples_leaf = 2, min_samples_split = 10 )
        # Fitting the data into the model
        decision_Tree.fit(X_cv, y)
        pkl_filename = "models/decision_Tree_model" + "_" +cat+ ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(decision_Tree, file)
    #Random Forest
    if cat == "general" or cat == "random forest":
        # Initializing the model
        random_forest = RandomForestClassifier(n_estimators=10, random_state=0)
        # Fitting the data into the model
        random_forest.fit(X_cv, y)
        pkl_filename = "models/random_forest_model" +"_"+ cat+".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(random_forest, file)
    #Stacking ( can only do if we trained on the other models)
    if cat == "general":
        estimators = [('naive bayes', naive_bayes),
              ('svm', svm),
              ('decision tree', decision_Tree)]

        # Initializing the model
        Stacking = StackingClassifier(estimators=estimators, final_estimator=logistic_regression)
        # Fitting the data into the model
        Stacking.fit(X_cv, y)
        pkl_filename = "models/stacking_model" + "_general"+ ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(Stacking, file)
