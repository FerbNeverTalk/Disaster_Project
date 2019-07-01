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
%matplotlib inline

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



class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        # tokenize by sentences
        sentence_list = sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            if tokenize(sentence) != []:
			
				pos_tags = pos_tag(tokenize(sentence))

            # index pos_tags to get the first word and part of speech tag
				first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
				if first_tag in ['VB', 'VBP'] or first_word == 'RT':
					return True

            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.starting_verb)

        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    engine = create_engine('sqlite:///figure8.db')
	df = pd.read_sql_table('figure8', engine)

	# drop irrelevant columns 
	df = df.drop(['id', 'original'], axis = 1)
	df = df[df.related != 2].reset_index(drop = True)


	# X is only the message column, Y is a multi-label binarizer with 36 classes
	X =  df['message']
	y =  df.drop(labels = 'message', axis = 1)
	
	return X,y,y.columns.values


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

	feature_union = FeatureUnion([
    ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
    ('verb_extract', StartingVerbExtractor())
	])


	pipeline = make_pipeline(feature_union, MultiOutputClassifier(RandomForestClassifier()))

	
	return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
	
	y_preds = model.predict(X_test)
	df_preds = pd.DataFrame(y_preds, columns = Y_test.columns.values)
	
	y_true = Y_test[category_names]
	y_pred = df_preds[category_names]
	
	print(f'The Classification Report --- \n\n {classification_report(y_true, y_pred)}\n')


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