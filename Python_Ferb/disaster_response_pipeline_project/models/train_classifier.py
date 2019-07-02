# import libraries
import sys

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords','averaged_perceptron_tagger'])

import pandas as pd
import numpy as np 
import re 
#import time
import pickle 
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from warnings import simplefilter
simplefilter(action = 'ignore', category = FutureWarning)



def load_data(database_filepath, threshold = 0.01):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    # drop irrelevant columns 
    df = df.drop(['id', 'original'], axis = 1)
    df = df[df.related != 2].reset_index(drop = True)
    
    # drop features that are stronly imbalance 
    df_temp = df.iloc[:,2:]
    dropped_dict = ((df_temp.sum(axis = 0)/len(df_temp) < threshold) |
                       (df_temp.sum(axis = 0)/len(df_temp) > 1 - threshold)).to_dict()
    
    dropped_columns = []
    for i in dropped_dict:
        if dropped_dict[i] == True:
            dropped_columns.append(i)
    
    df = df.drop(labels = dropped_columns, axis = 1)
    # X is only the message column, Y is a multi-label binarizer with 36 classes
    X =  df['message']
    y =  df.drop(labels = ['message','genre'], axis = 1)
    
    return X, y, y.columns.values

def tokenize(text):
    # normalize text: remove punctuation and lower case 
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # split text into word tokens 
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatizer all element in tokens and remove stopwords
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    
    return clean_tokens



def build_model():

    pipeline = make_pipeline(
        CountVectorizer(tokenizer = tokenize),
        TfidfTransformer(),
        MultiOutputClassifier(RandomForestClassifier())
    )
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    

    y_preds = model.predict(X_test)
    df_preds = pd.DataFrame(y_preds, columns = Y_test.columns.values)
    
    for i in category_names:
        y_true = Y_test[i]
        y_pred = df_preds[i]
        print(f'The Classification Report for feature: {i} --- \n\n {classification_report(y_true, y_pred)}\n')


def save_model(model, model_filepath):
    f = open(model_filepath, 'wb')
    pickle.dump(model, f)
    f.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()