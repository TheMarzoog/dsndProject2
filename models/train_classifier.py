import re
import sys
import nltk
import pickle
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


nltk.download(['punkt', 'stopwords', 'wordnet'])

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - file path of .db file

    OUTPUT:
    X - dataframe contaning the messages
    Y - dataframe contaning the labels
    catetory_names - list of labels names
    '''

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('tweets', engine)
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    INPUT:
    text - string of the orginal messege
    OUTPUT:
    text - string of the message after normalizing, tokenzing, removing stop words, and lemmatizing
    '''

    # normalize
    text = re.sub('\W', ' ', text.lower())
    # tokenize
    text = word_tokenize(text)
    # remove stop words
    text = [w for w in text if w not in stopwords.words('english')]
    # lemmatize
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    
    return text


def build_model():
    '''
    OUTPUT:
    model - grid search object
    '''

    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__bootstrap': [True, False],
        'clf__estimator__n_jobs': [1, 2, 3, 4, 5]
    } 

    cv = 5
    
    model = GridSearchCV(pipeline,
                        parameters,
                        cv=cv)
    
    
    return model


def evaluate_model(model, X_test, Y_test):
    '''
    compute Y_pred then print the evalusation of the model
    INPUT:
    model - grid search object
    X_test - test dataframe
    Y_test - dataframe of test labels
    '''
    
    Y_pred = model.predict(X_test)
    
    for i, col in enumerate(Y_test):
        print(col, classification_report(Y_test[col], Y_pred[:, i]))
              


def save_model(model, model_filepath):
    '''
    INPUT:
    model - estemator object to be saved
    model_filepath - string of the filepath
    '''
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    


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
        evaluate_model(model, X_test, Y_test)

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