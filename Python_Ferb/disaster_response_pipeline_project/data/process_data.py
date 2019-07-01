import sys

# import libraries
import pandas as pd 
import numpy as np 
import langid
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    
	messages = pd.read_csv(messages_filepath)
	categories = pd.read_csv( categories_filepath)
	
	df = pd.merge(messages, categories, on = 'id')
	
	return df 

def clean_data(df, threshold = 0.01):
    
	# separation 
	categories = df.categories.str.split(';', expand = True)
	
	# select the first row of the categories dataframe
	row = categories.iloc[0,:]
	# use this row to extract a list of new column names for categories.
	# one way is to apply a lambda function that takes everything 
	# up to the second to last character of each string with slicing
	category_colnames = row.apply(lambda x: x[:-2])
	
	categories.columns = category_colnames
	
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].apply(lambda x: x[-1])
		
		# convert column from string to numeric
		categories[column] = categories[column].astype('int')
	
	# drop the original categories column from `df`
	df = df.drop(labels = 'categories', axis = 1)
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, categories], axis = 1)
	
	# drop duplicates
	df.drop_duplicates(inplace = True)
	
	
	# Drop incorrect data 
	
	# Change genre as dummy 
	#df.genre[df.genre != 'direct'] = 0 
	#df.genre[df.genre == 'direct'] = 1 
	#df.rename(columns = {'genre':'genre_direct'}, inplace = True)
	#df.genre_direct = df.genre_direct.astype('int')

	
	# Create language variable
	#df['lang'] = ''
	
	#for ind, value in enumerate(df.original):
	#	try:
	#		df['lang'][ind] = langid.classify(value)[0]
	#	except:
	#		df['lang'][ind] = None

	
	# One-hot Encoding 
	#lang = pd.get_dummies(df['lang'], prefix = 'lang', dummy_na = True, drop_first = True)
	# Drop the original lang column from `df`
	#df = df.drop(labels = 'lang', axis = 1)
	# Concatenate the original dataframe with the new `lang` dataframe 
	#df = pd.concat([df, lang], axis = 1)
	
	# Drop the irrelevant columns 
	
	
	# dropped columns that have strong imbalancing data 
	df_temp = df.iloc[:,2:]
	dropped_dict = ((df_temp.sum(axis = 0)/len(df_temp) < threshold) | 
                    (df_temp.sum(axis =0)/len(df_temp) > 1-threshold)).to_dict()
	
	dropped_columns = []
    
    for i in dropped_dict:
        if dropped_dict[i] == True:
            dropped_columns.append(i)
			
	df_clean = df.drop(labels = dropped_columns, axis = 1)
	
	
	return df_clean
	
	
def save_data(df, database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
	df.to_sql('DisasterResponse', engine, index=False, if_exists = 'replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()