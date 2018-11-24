# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 08:36:35 2018

@author: hassans
"""

import pandas as pd

#from sklearn import model_selection
import pickle
import json


#from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score





#%% load the data
def load_data():
    train_data = pd.read_csv("C:/Local/hassans/Data/resumes/futurice/assignment/housing.csv")
    #print(train_data.head())
    
    # ignore column names/headers
    array = train_data.values
    
    # data to work with
    X = array[:,0:5]
    Y = array[:,5]
    
    return train_data, X, Y

   
    
#%% # Data scaling and splitting

def preprocess_data(data):
    
    convert = preprocessing.StandardScaler() 
    
    # separate input and output variables
    feature = data.drop(['house_value'], axis=1)
    label = data.house_value
    
    
    # preprocess the data
    featureT = convert.fit_transform(feature.values)
    #labelT = convert.fit_transform(data.house_value.values.reshape(-1,1)).flatten()  
    labelT = label
    return convert, featureT, label
    
def split_data(featureT, labelT):
    
    # separate train and test data
    test_size = 0.33
    seed = 7
    feature_train, feature_test, label_train, label_test = train_test_split(featureT, labelT, test_size = test_size, random_state = seed)
    
    return feature_train, feature_test, label_train, label_test
    
    
    
    
    
#%% train and test with random forest regressor

def random_forest_train(feature_train, feature_test, label_train, label_test):
    # train the model with Random Forest Regressor
    seed = 7
    forest_reg = RandomForestRegressor(random_state = seed)
    # fit the train data
    forest_reg.fit(feature_train,label_train)
    # r2 score for training data
    forest_train_r2_score = r2_score(forest_reg.predict(feature_train),label_train)
    print(forest_train_r2_score)
    
    # 10 fold cross validation
    forest_cv10_score = cross_val_score(forest_reg, feature_train, label_train, cv=10)
    print(forest_cv10_score.mean())
    
    
    # let's see how well the random forest regressor fits well with the test data
    forest_score = r2_score(forest_reg.predict(feature_test),label_test) 
    print(forest_score)
    
    
    
#%% Final training with random forest regressor

def final_model_train(data):
    
    convert, X, y = preprocess_data(data)
    print (convert.mean_, convert.scale_)
    
    seed = 7
    
    model = RandomForestRegressor(random_state = seed)
    model.fit(X, y)
    
    model_r2_score = r2_score(model.predict(X), y)
    model_cv_score = cross_val_score(model, X, y, cv = 10)
    
    print(model_r2_score)
    print(model_cv_score)
    
    # save model and converter to disk
    
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))    

    filename = 'converter.sav'
    pickle.dump(convert, open(filename, 'wb'))    

    
    
    
    
    
                           
#%% main functionalities
    
#data, X, y = load_data()
#convert, X, y = preprocess_data(data)
#X_train, X_test, y_train, y_test = split_data(X, y)
#train_model(X_train, y_train)    
#random_forest_train(X_train, X_test, y_train, y_test)
#final_model_train(data)

# load the save model and converter

#%% load data and train model
def train_model():
    data, X, y = load_data()
    
    # check the model train
    convert, X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    random_forest_train(X_train, X_test, y_train, y_test)
    
    final_model_train(data)
    
    
#%% test valid data
def valid_data(request):
    
    #value_to_predict = json.loads(request.body) # deserialize object :P
    #print(value_to_predict)
    
    
    
    #print(type(value_to_predict))
    
    data = pd.read_json(request.body, typ= 'series', orient = 'records')
    dataF = data.to_frame().T # series to dataframe and then make the first one as column name
    
    col_names = ['crime_rate', 'avg_number_of_rooms', 'distance_to_employment_centers', 'property_tax_rate', 'pupil_teacher_ratio']
    
    # if all the columns exists?
    col_exists = pd.Series(col_names).isin(dataF.columns).all()
    print(col_exists)
    
    if col_exists:
        result = predict(dataF)
        return result
    else:
        return col_exists
    
    
    
    #print(test)
    
    
    #for key, value in value_to_predict.items():
     #   print (key, value)
        
            
#%% Predict the data
def predict(X):
    
    #print(X.shape)
    # load the converter
    filename = 'converter.sav'
    convert = pickle.load(open(filename, 'rb'))
    #print('filenfound')
    
    #print(convert.mean_)
    #print(convert.scale_)
    

    featureT = convert.transform(X.values)
    #print(featureT)
    
    # load the model
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    #print(result)
    
    #print(model)
    
    yHat = model.predict(featureT)
    #print(yHat)
    
    
    return yHat
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
